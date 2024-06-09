from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredHTMLLoader
from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
import torch
import transformers
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def read_docs(folder_path):
    loader = DirectoryLoader(folder_path, glob="*.html", show_progress=True, use_multithreading=False, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    return docs


def chunk(docs, headers, chunk_type="Recursive", **kwargs):

    html_splitter = HTMLSectionSplitter(headers)
    html_header_splits = html_splitter.split_documents(docs)

    match chunk_type:
        case "Recursive":
            split_doc = recursive_split(splits=html_header_splits, **kwargs)
        case _:
            split_doc = recursive_split(splits=html_header_splits, **kwargs)

    return split_doc


def recursive_split(splits, separators=None, chunk_size=800, chunk_overlap=100):
    separators = ["\n\n", "\n", "(?<=\. )", " ", ""]
    rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                       separators=separators)

    recursive_header_split = rec_char_splitter.split_documents(splits)

    return recursive_header_split


def create_vector_index_and_embedding_model(chunks):
    store = LocalFileStore("./cache/")

    embed_model_id = 'intfloat/e5-small-v2'
    model_kwargs = {"device": "cpu", "trust_remote_code": True}

    embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

    embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace=embed_model_id)

    vector_index = FAISS.from_documents(chunks, embedder)

    return embeddings_model, vector_index


def retrieve_top_k_docs(query, vector_index, embedding_model, k=4):
    query_embedding = embedding_model.embed_query(query)
    docs = vector_index.similarity_search_by_vector(query_embedding, k=k)

    return docs


def generate_model(model_id):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, config=model_config,
                                                              quantization_config=bnb_config, device_map="auto")

    # Set the model in evaluation stage since we need to perform inference
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def create_pipeline(model, tokenizer, temperature, repetition_penalty, max_new_tokens):

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )

    llm_pipeline = HuggingFacePipeline(pipeline=pipeline)

    return llm_pipeline


def create_qa_RAG_chain(llm_pipeline, retriever, system_prompt):
    # https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                               ("human", "{input}")])

    qa_chain = create_stuff_documents_chain(llm_pipeline, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)

    return chain


def create_qa_RAG_chain_history(llm_pipeline, retriever, system_prompt):
    """
    Creates a chain exploiting also the history of the LLM
    :param llm_pipeline:
    :param retriever: This should be an history aware retriever
    :param system_prompt:
    :return:
    """
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                  MessagesPlaceholder("chat_history"),
                                                  ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def create_rephrase_history_chain(llm_pipeline, retriever, system_prompt):
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                               MessagesPlaceholder("chat_history"),
                                                               ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm_pipeline, retriever, contextualize_q_prompt)

    return history_aware_retriever


def get_answer_RAG(qa_chain, question):
    answer = qa_chain.invoke({"input": question})["answer"]
    return answer


def answer_LLM_only(model, tokenizer, query):
    """
    Answers a question by using only the knowledge contained inside the model, without using RAG.
    The query in input will be executed as-is. If the model requires some tokens for instruction tuning, they must be included already when passing the query in input.
    :param model: the model to use for generating the answer
    :param tokenizer: tokenizer for the model
    :param query: the query to be run
    :return: the answer of the llm
    """

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    query_tokenized = tokenizer.encode_plus(query, return_tensors="pt")["input_ids"].to('cuda')
    answer_ids = model.generate(query_tokenized,
                                max_new_tokens=256,
                                do_sample=True)

    decoded_answer = tokenizer.batch_decode(answer_ids)

    return decoded_answer


def create_conversational_chain(rag_chain):
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain