from fastapi import FastAPI, Query
from typing import Optional, List
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import boilerplate as B
import logging
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import maude_search as MS
import LLMService as LM
import uuid
import time
from threading import Lock
import pandas as pd
import os

logger = logging.getLogger(__name__)

app = FastAPI()
stop_dict = {}
stop_lock = Lock()
status_dict = {}
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/downloads", StaticFiles(directory="maude output"), name="downloads")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/maude/maude-styling.css", response_class=FileResponse)
def styling():
    return FileResponse("static/maude-styling.css")

@app.get("/favicon.ico", response_class=FileResponse, include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")

@app.post("/maude/status/update")
def update_status(id: str, status:str):
    status_dict[id] = status

@app.get("/maude/status/retrieve")
def retrieve_status(id: Optional[str] = None):
    if id == None:
        return "Bottom Text"

    return status_dict.get(id, "Sending query...")

@app.get("/maude/recover", response_class=HTMLResponse)
def recover_results(recover: Optional[str] = None):
    if recover == None or recover == "":
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Search Results</h2>
            Please enter a valid search ID!
            </div>
            <a href="/maude/" class="btn"> <br> Back to Search</a>

        </body>
        </html>
        """
    elif len(recover) > 0:
        try:

            def add_download_links(df, query):
                df = df.copy()
                def linkify(label):
                    href = f"/downloads/{recover}/{query}_{label}.csv"
                    return f'<a href="{href}" download>Download CSV</a>'
                df["Download"] = df["Cluster"].apply(linkify)
                return df
            files = sorted(os.listdir(f"maude output/{recover}"))
            query = files[1].split("_")[0]

            df = pd.read_csv(f"maude output/{recover}/!summary_{recover}.csv", usecols=[1,2,3,4,5,6])
            df = add_download_links(df, query)
            df = df.to_html(escape=False, index=False)
            href_summary = f"/downloads/{recover}/!summary_{recover}.csv"
            html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Recovered Search Results</h2>
            <a href="{href_summary}" download>Download Summary CSV</a> <br>
            <a href="/maude/" class="btn"> <br> Back to Search</a>
            {df}
            </div>


        </body>
        </html>
        """

        except Exception as e:
            print(e)
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
            <link rel="stylesheet" href="/static/results_styling.css">
                <title>Recovered Search Results</title>
            </head>
            <body>
                <div class="container mt-4">
                <h2>Search Results</h2>
                Search ID not found!
                </div>
                <a href="/maude/" class="btn"> <br> Back to Search</a>

            </body>
            </html>
            """


    return HTMLResponse(html_content)


@app.get("/maude/search", response_class=HTMLResponse)
def search(
    query: Optional[str] = None,
    brand: Optional[str] = None,
    manu: Optional[str] = None,
    model_num: Optional[str] = None,
    code: Optional[str] = None,
    must: Optional[str] = None,
    must2: Optional[str] = None,
    must_operator: Optional[str] = None,
    search_type: Optional[bool] = False,
    years: Optional[List[int]] = Query(default=[]),
    search_id: Optional[str] = None
):
    if query == None or query == "":
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Search Results</h2>
            Please enter a valid search!
            </div>
            <a href="/maude/" class="btn"> <br> Back to Search</a>

        </body>
        </html>
        """
        return HTMLResponse(html_content)
    
    if len(years) == 0:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Search Results</h2>
            Please select at least one year!
            </div>
            <a href="/maude/" class="btn"> <br> Back to Search</a>

        </body>
        </html>
        """
        return HTMLResponse(html_content)
    
        
    logging.info(f"Search: {query}")
    logging.info(f"Search ID: {search_id}")

    df, num_clusters, total = MS.perform_search(search_id, service, query, brand, manu, model_num, code, must, must2, must_operator, search_type, years, stop_dict, stop_lock, status_dict)
    if isinstance(df, str):
        if query == "":
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
            <link rel="stylesheet" href="/static/results_styling.css">
                <title>Search Results</title>
            </head>
            <body>
                <div class="container mt-4">
                <h2>Search Results</h2>
                Please enter a valid search!
                </div>
                <a href="/maude/" class="btn"> <br> Back to Search</a>

            </body>
            </html>
            """
        else:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
            <link rel="stylesheet" href="/static/results_styling.css">
                <title>Search Results</title>
            </head>
            <body>
                <div class="container mt-4">
                <h2>Search Results</h2>
                No results found!
                </div>
                <a href="/maude/" class="btn"> <br> Back to Search</a>

            </body>
            </html>
            """
            
        
    elif df is None or df.empty:
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Search Results</h2>
            No results found!
            </div>
            <a href="/maude/" class="btn"> <br> Back to Search</a>

        </body>
        </html>
        """
    else:
        def add_download_links(df, query):
            df = df.copy()
            def linkify(label):
                href = f"/downloads/{search_id}/{query}_{label}.csv"
                return f'<a href="{href}" download>Download CSV</a>'
            df["Download"] = df["Cluster"].apply(linkify)
            return df
        df_with_links = add_download_links(df, query)
        html_table = df_with_links.to_html(escape=False, index=False)
        href_summary = f"/downloads/{search_id}/!summary_{search_id}.csv"


        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <link rel="stylesheet" href="/static/results_styling.css">
            <title>Search Results</title>
        </head>
        <body>
            <div class="container mt-4">
            <h2>Search Results: {query}</h2>
        
            <details>
            <summary>Detailed Search Parameters:</summary>
            <b>Device Name:</b> {brand} <br>
            <b>Manufacturer:</b> {manu} <br>
            <b>Model Number:</b> {model_num} <br>
            <b>Device Code:</b> {code} <br>
            <b>Must Contain:</b> {must} {must_operator} {must2} <br>
            <b>Detailed Search:</b> {search_type} <br>
            <b>Years Searched:</b> {years}
            </details>
       <br>
            <div style="font-size: 14px; font-weight: normal;">Search ID: {search_id}</div> <br>
            <a href="{href_summary}" download>Download Summary CSV</a> <br> <br>

            <div style="font-size: 18px; font-weight: bold;">Clusters: {num_clusters}</div> 
            <div style="font-size: 16px; font-weight: bold;">Number of entries: {total}</div>


            <a href="/maude/" class="btn"> <br> Back to Search</a>
            <br>

            {html_table}
            </div>


        </body>
        </html>
        """
        with stop_lock:
            if stop_dict.get(search_id):
                stop_dict.pop(search_id)
    return HTMLResponse(content=html_content) 

@app.get("/maude/cancel")
def stop_search(search_id: str):
    with stop_lock:
        if search_id in stop_dict:
            stop_dict[search_id] = True
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
        <link rel="stylesheet" href="/static/results_styling.css">
        <title>Search Cancelled!</title>
        </head>
        <body>
        <h1>Search Cancelled!</h1>
        <a href="/maude/" class="btn"> <br> Back to Search</a>
        </body>

        </html>
"""
        return HTMLResponse(content=html_content)


@app.get("/maude/", response_class=HTMLResponse)
def landing():
    strout = B.HEADER.format(TITLE = "MAUDE search")
    search_id = str(uuid.uuid4())
    with stop_lock:
        stop_dict[search_id] = False
    strout += B.SEARCH_FORM.format(search_id=search_id)
    strout += B.FOOTER
    return strout

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        filename="logs/" + str(time.asctime().replace(":","-")) + ".txt",
                        filemode="a+", format="%(asctime)-15s %(levelname)-9s %(message)s")
    logging.info("Logging intiialized successfully")  
    service = LM.LLMService(LM.LLMServiceConfig)
    logging.info("LLM service loaded successfully")
    
    HOST = "10.10.205.17"
    PORT = 8001
    logging.info(f"Setting IO to {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
    logging.info("App service started by Uvicorn")

