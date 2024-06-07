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
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StdOutCallbackHandler


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

    return model, tokenizer

def create_pipeline(model, tokenizer):

    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        return_full_text=True,
        temperature=0.5,
        max_new_tokens=256
    )

    llm_pipeline = HuggingFacePipeline(pipeline=pipeline)

    return llm_pipeline


def create_qa_RAG_chain(llm_pipeline, retriever, system_prompt):

    # https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

    qa_chain = create_stuff_documents_chain(llm_pipeline, prompt)
    chain = create_retrieval_chain(retriever, qa_chain)

    return chain


def get_answer_RAG(qa_chain, question):
    answer = qa_chain.invoke({"input": question})["answer"]
    return answer