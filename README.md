## Modifications Made In This Repo
- Forked from [ollama-deep-researcher](https://github.com/langchain-ai/ollama-deep-researcher).
- Added support for searching with `SearXNG`
  - Requires `SearXNG` to be running locally. 
  - See [SearXNG-Docker](https://github.com/searxng/searxng-docker) for instructions on creating a SearXNG instance using Docker.
  - When you install SearXNG, the only active output format by default is the HTML format.
    You need to activate the `json` format to use the API. This can be done by adding the following line to the `settings.yml` file:
    ```yaml
    search:
        formats:
            - html
            - json
    ```
- Added support for LLM inferencing with OpenAI compatible APIs.
  - Supported LLM providers:
    - OpenAI
    - Gemini
    - DeepSeek
    - Ollama (local)
  - API keys already included in the `.env` file.

- Potential future optimizations:
  - Modify the prompt in `src\assistant\prompts.py` to improve the performance on finance related content.
  - Adjust the parameters in function `searxng_search` (`src\assistant\utils.py`) for better search results. 
    - Specify websites to search from.


## Running with Docker

Clone the repo and build an image:
```
$ docker build -t ollama-deep-researcher .
```

Run the container:
```
$ docker run --rm -it -p 2024:2024 ollama-deep-researcher
```

NOTE: You will see log message:
```
2025-02-10T13:45:04.784915Z [info     ] 🎨 Opening Studio in your browser... [browser_opener] api_variant=local_dev message=🎨 Opening Studio in your browser...
URL: https://smith.langchain.com/studio/?baseUrl=http://0.0.0.0:2024
```
...but the browser will not launch from the container.

Instead, visit this link with the correct baseUrl IP address: [`https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024`](https://smith.langchain.com/studio/thread?baseUrl=http://127.0.0.1:2024)

### Using the LangGraph Studio UI

When you launch LangGraph server, you should see the following output and Studio will open in your browser:
> Ready!
>
> API: http://127.0.0.1:2024
>
> Docs: http://127.0.0.1:2024/docs
>
> LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Open `LangGraph Studio Web UI` via the URL in the output above.