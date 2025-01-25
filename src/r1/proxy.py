import json
import re
from pprint import pprint

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()

# Create a persistent client with longer timeout - 5 minutes for LLM response time
client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    # Forward request to TGI running on port 8001
    url = f"http://localhost:8001/{path}"

    # Forward all headers except host, add TGI auth
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove host to avoid conflicts
    headers["Authorization"] = "Bearer fake-api-key"  # Required by TGI

    # Forward the request with same method and body
    request_body = await request.body()
    edited_rbody = json.loads(request_body.decode())
    edited_rbody["temperature"] = 0.6
    new_req_body = json.dumps(edited_rbody).encode()
    headers["content-length"] = str(len(new_req_body))
    response = await client.request(
        method=request.method, url=url, content=new_req_body, headers=headers
    )

    # For chat completions, clean <think> tags from response
    if path.endswith("/chat/completions") and response.status_code == 200:
        data = response.json()
        print("The request body:")
        pp_req_body = json.dumps(json.loads(new_req_body.decode()), indent=2)
        print(GREEN + pp_req_body + RESET)
        print("The response:")
        print(RED + json.dumps(data, indent=2) + RESET)
        # if 'choices' in data:
        #     for choice in data['choices']:
        #         if 'message' in choice and 'content' in choice['message']:
        #             choice['message']['content'] = re.sub(
        #                 r'<think>.*?</think>',
        #                 '',
        #                 choice['message']['content'],
        #                 flags=re.DOTALL
        #             ).strip()
        return data

    # For all other endpoints/responses, proxy as-is
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


# Clean up the client on shutdown
@app.on_event("shutdown")
async def shutdown():
    await client.aclose()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
