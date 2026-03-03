**REMINDERS**
DONE: 1. Refactor data model and relationship, do not mix SQLModel with FastAPI Request/Response models are 2 different things, Model is defined and semantically correct, while view adapt to read what necessary from the model and update what necessary
DONE: 2. Refactor Adapters to be self-contained and configured once then acting as a processing pipeline at every model response. The mlx_server should just throw at them the raw model response and the Adapters do the rest. They should be used for all models so that from the Server perspective all models are equal, they receive a routed request, they generate an output, deliver it to the adapter which produces a valid and protocol compliant response (probably which should have Adapter + Protocol (Anthropic or OpenAI))
DONE: 3. Update probing process to leverage new Adapters infrastructure
DONE: 4. LiquidAI thinking not recognized properly
DONE: 5. Refactor UI component to consistently read from the models, using the right properties, and add switch for API kind to Chat (Anthropic vs OpenAI).
DONE: 6. Profile option to load at startup doesn't work
DONE: 7. In the settings page we need to change the "Model Pool" concept to "Execution Profiles Pool" or similar, as we have changed that concept. This include shaving a pre-loaded profile and the list should show profiles not models
8. DevStral (Mistral) family adapter is not recognizing Devstral models correctly
9. Cancel button on started downloads, pause/resume as well
