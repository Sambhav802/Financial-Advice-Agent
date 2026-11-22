from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import build_app, ticker_check

MODEL = "gpt-oss-120b"

graph_app = build_app(MODEL, override_env=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    user_query: str = ""


# Sync endpoint: FastAPI runs it in a threadpool, so the blocking graph call
# doesn't stall the event loop.
@app.post("/analyze")
def analyze(body: AnalyzeRequest):
    try:
        if not body.user_query:
            return JSONResponse({"error": "user_query is required"}, status_code=400)

        state = graph_app.invoke({"user_query": body.user_query})
        response_data = {"final_response": state["final_response"][-1].content}

        if ticker_check(state) == "yes":
            report = state.get("final_report")
            response_data.update(
                {
                    "price_analyst_report": state.get("price_analyst_report", ""),
                    "news_analyst_report": state.get("news_analyst_report", ""),
                    "final_report": report.model_dump() if report else None,
                }
            )

        return response_data

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=5000, reload=True)
