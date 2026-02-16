DONE: 1. Refactor data model and relationship, do not mix SQLModel with FastAPI Request/Response models are 2 different things, Model is defined and semantically correct, while view adapt to read what necessary from the model and update what necessary
IN PROGRESS: 2. Refactor UI component to consistently read from the models, using the right properties, and add switch for API kind to Chat (Anthropic vs OpenAI). Only text and vision models should have the chat button the other not. Tool use shouldn't go via prompt injection but use the model adapter as to test production behavior
DONE: 3. Refactor Adapters to be self-contained and configured once then acting as a processing pipeline at every model response. The mlx_server should just throw at them the raw model response and the Adapters do the rest. They should be used for all models so that from the Server perspective all models are equal, they receive a routed request, they generate an output, deliver it to the adapter which produces a valid and protocol compliant response (probably which should have Adapter + Protocol (Anthropic or OpenAI))
DONE: 4. Update probing process to leverage new Adapters infrastructure
IN PROGRESS: 5. LiquidAI thinking not recognized properly
6. DevStral (Mistral) family adapter is not recognizing Devstral models correctly
7. Cancel button on started downloads, pause/resume as well
8. Settings are broken need to be updated possibly using tabs because the page is getting very long especially with the providers configuration and the logging which should definitely have its own page
