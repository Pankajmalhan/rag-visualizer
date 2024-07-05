import os
from dotenv import load_dotenv
import re
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import umap

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

with open("chunks.pickle", "rb") as f:
    chunks = pickle.load(f)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

with open("embedding.pickle", "rb") as f:
    all_embeddings = pickle.load(f)


pca = PCA(n_components=2)
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)

app = Dash(__name__)

app.layout = html.Div([
    html.H1("RAG Data Visualization", style={'textAlign': 'center'}),
    html.Div([
        dcc.Input(
            id="question-input",
            type="text",
            placeholder="Enter your question here",
            style={'width': '60%', 'height': '40px', 'fontSize': '16px', 'marginRight': '10px'}
        ),
        dcc.Dropdown(['PCA', 'T-SNE', 'UMAP'], 'PCA', id='algo', style={'width': '100px', 'height': '40px', 'fontSize': '16px', 'marginRight': '10px'}),
        html.Button(
            'Submit',
            id='submit-button',
            n_clicks=0,
            style={'height': '40px', 'fontSize': '16px'}
        )
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    html.Hr(),
    html.Div(id="answer-output", style={'marginBottom': '20px', 'textAlign': 'center', 'fontSize': '18px'}),
    dcc.Graph(id="embedding-plot")
])

@app.callback(
    [Output("answer-output", "children"),
     Output("embedding-plot", "figure")],
    [Input("submit-button", "n_clicks")],
    [Input("algo", "value")],
    [State("question-input", "value")]
)
def update_output(n_clicks,algo, question):
    if n_clicks > 0 and question:
        answer = qa_chain.run(question)
        
        docs = retriever.invoke(question)
        doc_texts = [doc.page_content for doc in docs]
        
        question_embedding = embeddings.embed_documents([question])[0]
        answer_embedding = embeddings.embed_documents([answer])[0]

        documents_count = len(all_embeddings)
        all_embeddings_temp = all_embeddings.copy()
        all_embeddings_temp.append(question_embedding)
        all_embeddings_temp.append(answer_embedding)

        if algo == "PCA":
            all_embeddings_temp_tran = pca.fit_transform(all_embeddings_temp)
            all_embeddings_2d = all_embeddings_temp_tran[:documents_count]
            qa_embeddings_2d = all_embeddings_temp_tran[documents_count:]
        elif algo == "T-SNE":
            all_embeddings_temp_tran = tsne.fit_transform(np.array(all_embeddings_temp))
            all_embeddings_2d = all_embeddings_temp_tran[:documents_count]
            qa_embeddings_2d = all_embeddings_temp_tran[documents_count:]
        elif algo == "UMAP":
            reducer = umap.UMAP()
            all_embeddings_temp_tran = reducer.fit_transform(np.array(all_embeddings_temp))
            all_embeddings_2d = all_embeddings_temp_tran[:documents_count]
            qa_embeddings_2d = all_embeddings_temp_tran[documents_count:]
            
        
        distances = euclidean_distances([question_embedding], all_embeddings)[0]
        print(distances)
        
        max_distance = np.max(distances)
        normalized_sizes = 1 - (distances / max_distance)
        point_sizes = normalized_sizes * 18 + 7
        
        df = pd.DataFrame(all_embeddings_2d, columns=['x', 'y'])
        df['type'] = 'Corpus'
        df['size'] = point_sizes
        df['text'] = chunks
        
        qa_df = pd.DataFrame(qa_embeddings_2d, columns=['x', 'y'])
        qa_df['type'] = ['Question', 'Answer']
        qa_df['size'] = 10
        qa_df['text'] = [question, answer]
        df = pd.concat([df, qa_df], ignore_index=True)
        
        retrieved_indices = [chunks.index(doc) for doc in doc_texts]
        df.loc[retrieved_indices, 'type'] = 'Retrieved'

        df['text'] = df['text'].str.wrap(30)
        df['text'] = df['text'].apply(lambda x: x.replace('\n', '<br>'))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Corpus']['x'],
            y=df[df['type'] == 'Corpus']['y'],
            mode='markers',
            name='Corpus',
            marker=dict(size=df[df['type'] == 'Corpus']['size'], color='red', opacity=0.5),
            text=df[df['type'] == 'Corpus']['text'],
            hoverinfo='text'
        ))
        
        fig.add_trace(go.Scatter(
            x=df[df['type'] == 'Retrieved']['x'],
            y=df[df['type'] == 'Retrieved']['y'],
            mode='markers',
            name='Retrieved',
            marker=dict(size=df[df['type'] == 'Retrieved']['size'] * 1, color='green'),
            text=df[df['type'] == 'Retrieved']['text'],
            hoverinfo='text'
        ))
        
        for t in ['Question', 'Answer']:
            fig.add_trace(go.Scatter(
                x=df[df['type'] == t]['x'],
                y=df[df['type'] == t]['y'],
                mode='markers',
                name=t,
                marker=dict(size=24, symbol='diamond'),
                text=df[df['type'] == t]['text'],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title="",
            height=800,
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            hovermode='closest'
        )
        
        return html.Div([
            html.Strong("Question: "), html.Span(question),
            html.Br(),
            html.Strong("Answer: "), html.Span(answer)
        ]), fig
    
    return "Ask a question and click Submit", go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)