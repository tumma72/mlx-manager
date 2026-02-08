We have an error with Nvidia Nemotron models:

2026-02-07 16:30:47 | DEBUG    | mlx_manager.mlx_server.models.pool:_estimate_model_size:171 - Model size from disk for mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit: 16.6GB weights + 5% overhead = 17.4GB
2026-02-07 16:30:47 | INFO     | mlx_manager.mlx_server.models.pool:_load_model:313 - Loading model: mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
2026-02-07 16:30:47 | DEBUG    | mlx_manager.mlx_server.models.detection:detect_model_type:40 - Loaded config for mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit: keys=['architectures', 'attention_bias', 'attention_dropout', 'auto_map', 'bos_token_id', 'chunk_size', 'conv_kernel', 'eos_token_id', 'expand', 'head_dim', 'hidden_dropout', 'hidden_size', 'hybrid_override_pattern', 'initializer_range', 'intermediate_size', 'layer_norm_epsilon', 'mamba_head_dim', 'mamba_hidden_act', 'mamba_num_heads', 'mamba_proj_bias', 'mamba_ssm_cache_dtype', 'max_position_embeddings', 'mlp_bias', 'mlp_hidden_act', 'model_type', 'moe_intermediate_size', 'moe_shared_expert_intermediate_size', 'n_group', 'n_groups', 'n_routed_experts', 'n_shared_experts', 'norm_eps', 'norm_topk_prob', 'num_attention_heads', 'num_experts_per_tok', 'num_hidden_layers', 'num_key_value_heads', 'num_logits_to_keep', 'pad_token_id', 'partial_rotary_factor', 'quantization', 'quantization_config', 'rescale_prenorm_residual', 'residual_in_fp32', 'rope_theta', 'routed_scaling_factor', 'sliding_window', 'ssm_state_size', 'tie_word_embeddings', 'time_step_floor', 'time_step_limit', 'time_step_max', 'time_step_min', 'topk_group', 'torch_dtype', 'transformers_version', 'use_bias', 'use_cache', 'use_conv_bias', 'use_mamba_kernels', 'vocab_size']
2026-02-07 16:30:47 | DEBUG    | mlx_manager.mlx_server.models.detection:detect_model_type:151 - Detected AUDIO model from name pattern: mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
2026-02-07 16:30:47 | INFO     | mlx_manager.mlx_server.models.pool:_load_model:328 - Detected model type: audio for mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
2026-02-07 16:30:47 | DEBUG    | logging:callHandlers:1762 - https://logfire-eu.pydantic.dev:443 "POST /v1/traces HTTP/1.1" 200 0
Fetching 12 files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 52428.80it/s]
Download complete: : 0.00B [00:00, ?B/s]              2026-02-07 16:30:47 | ERROR    | mlx_manager.mlx_server.models.pool:_load_model:401 - Failed to load model mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit: Could not determine model type for mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
Traceback (most recent call last):

  File "<string>", line 1, in <module>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               │     │   └ 4
               │     └ 7
               └ <function _main at 0x105b83100>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/multiprocessing/spawn.py", line 135, in _main
    return self._bootstrap(parent_sentinel)
           │    │          └ 4
           │    └ <function BaseProcess._bootstrap at 0x105a9f1a0>
           └ <SpawnProcess name='SpawnProcess-1' parent=1532 started>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
    │    └ <function BaseProcess.run at 0x105a9e700>
    └ <SpawnProcess name='SpawnProcess-1' parent=1532 started>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
    │    │        │    │        │    └ {'config': <uvicorn.config.Config object at 0x10af612e0>, 'target': <bound method Server.run of <uvicorn.server.Server object...
    │    │        │    │        └ <SpawnProcess name='SpawnProcess-1' parent=1532 started>
    │    │        │    └ ()
    │    │        └ <SpawnProcess name='SpawnProcess-1' parent=1532 started>
    │    └ <function subprocess_started at 0x10afed760>
    └ <SpawnProcess name='SpawnProcess-1' parent=1532 started>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/uvicorn/_subprocess.py", line 80, in subprocess_started
    target(sockets=sockets)
    │              └ [<socket.socket fd=3, family=2, type=1, proto=0, laddr=('127.0.0.1', 10241)>]
    └ <bound method Server.run of <uvicorn.server.Server object at 0x10b17a060>>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/uvicorn/server.py", line 67, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
           │           │    │             │                      │    │      └ <function Config.get_loop_factory at 0x10af4fba0>
           │           │    │             │                      │    └ <uvicorn.config.Config object at 0x10af612e0>
           │           │    │             │                      └ <uvicorn.server.Server object at 0x10b17a060>
           │           │    │             └ [<socket.socket fd=3, family=2, type=1, proto=0, laddr=('127.0.0.1', 10241)>]
           │           │    └ <function Server.serve at 0x10afec860>
           │           └ <uvicorn.server.Server object at 0x10b17a060>
           └ <function run at 0x105db5120>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
           │      │   └ <coroutine object Server.serve at 0x10b16d8c0>
           │      └ <function Runner.run at 0x105e0d580>
           └ <asyncio.runners.Runner object at 0x10af63380>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           │    │     │                  └ <Task pending name='Task-1' coro=<Server.serve() running at /Users/atomasini/Development/mlx-manager/backend/.venv/lib/python...
           │    │     └ <cyfunction Loop.run_until_complete at 0x10b1996c0>
           │    └ <uvloop.Loop running=True closed=False debug=False>
           └ <asyncio.runners.Runner object at 0x10af63380>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/uvicorn/protocols/http/httptools_impl.py", line 416, in run_asgi
    result = await app(  # type: ignore[func-returns-value]
                   └ <uvicorn.middleware.proxy_headers.ProxyHeadersMiddleware object at 0x10b3a0710>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
                 │    │   │      │        └ <bound method RequestResponseCycle.send of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d95700>>
                 │    │   │      └ <bound method RequestResponseCycle.receive of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d957...
                 │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
                 │    └ <fastapi.applications.FastAPI object at 0x10f467ce0>
                 └ <uvicorn.middleware.proxy_headers.ProxyHeadersMiddleware object at 0x10b3a0710>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/fastapi/applications.py", line 1138, in __call__
    await super().__call__(scope, receive, send)
                           │      │        └ <bound method RequestResponseCycle.send of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d95700>>
                           │      └ <bound method RequestResponseCycle.receive of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d957...
                           └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
          │    │                │      │        └ <bound method RequestResponseCycle.send of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d95700>>
          │    │                │      └ <bound method RequestResponseCycle.receive of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d957...
          │    │                └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <starlette.middleware.errors.ServerErrorMiddleware object at 0x10f6f9a00>
          └ <fastapi.applications.FastAPI object at 0x10f467ce0>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
          │    │   │      │        └ <function ServerErrorMiddleware.__call__.<locals>._send at 0x129d0c0e0>
          │    │   │      └ <bound method RequestResponseCycle.receive of <uvicorn.protocols.http.httptools_impl.RequestResponseCycle object at 0x129d957...
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <opentelemetry.instrumentation.asgi.OpenTelemetryMiddleware object at 0x10f6f9910>
          └ <starlette.middleware.errors.ServerErrorMiddleware object at 0x10f6f9a00>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/opentelemetry/instrumentation/asgi/__init__.py", line 810, in __call__
    await self.app(scope, otel_receive, otel_send)
          │    │   │      │             └ <function ServerErrorMiddleware.__call__.<locals>._send at 0x11b3b60c0>
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <starlette.middleware.errors.ServerErrorMiddleware object at 0x10f6f9790>
          └ <opentelemetry.instrumentation.asgi.OpenTelemetryMiddleware object at 0x10f6f9910>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/middleware/errors.py", line 164, in __call__
    await self.app(scope, receive, _send)
          │    │   │      │        └ <function ServerErrorMiddleware.__call__.<locals>._send at 0x123fd16c0>
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app.<locals>.build_middleware_stack.<locals>.ExceptionH...
          └ <starlette.middleware.errors.ServerErrorMiddleware object at 0x10f6f9790>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/opentelemetry/instrumentation/fastapi/__init__.py", line 307, in __call__
    await self.app(scope, receive, send)
          │    │   │      │        └ <function ServerErrorMiddleware.__call__.<locals>._send at 0x123fd16c0>
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <starlette.middleware.cors.CORSMiddleware object at 0x10f606ab0>
          └ <opentelemetry.instrumentation.fastapi.FastAPIInstrumentor.instrument_app.<locals>.build_middleware_stack.<locals>.ExceptionH...
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/middleware/cors.py", line 93, in __call__
    await self.simple_response(scope, receive, send, request_headers=headers)
          │    │               │      │        │                     └ Headers({'host': 'localhost:10241', 'accept': '*/*', 'content-type': 'application/json', 'origin': 'http://localhost:5173', '...
          │    │               │      │        └ <function ServerErrorMiddleware.__call__.<locals>._send at 0x123fd16c0>
          │    │               │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │               └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <function CORSMiddleware.simple_response at 0x10cf88b80>
          └ <starlette.middleware.cors.CORSMiddleware object at 0x10f606ab0>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/middleware/cors.py", line 144, in simple_response
    await self.app(scope, receive, send)
          │    │   │      │        └ functools.partial(<bound method CORSMiddleware.send of <starlette.middleware.cors.CORSMiddleware object at 0x10f606ab0>>, sen...
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <starlette.middleware.exceptions.ExceptionMiddleware object at 0x10f6f9970>
          └ <starlette.middleware.cors.CORSMiddleware object at 0x10f606ab0>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
          │                            │    │    │     │      │        └ functools.partial(<bound method CORSMiddleware.send of <starlette.middleware.cors.CORSMiddleware object at 0x10f606ab0>>, sen...
          │                            │    │    │     │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │                            │    │    │     └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │                            │    │    └ <starlette.requests.Request object at 0x129d97a40>
          │                            │    └ <fastapi.middleware.asyncexitstack.AsyncExitStackMiddleware object at 0x10f6f9670>
          │                            └ <starlette.middleware.exceptions.ExceptionMiddleware object at 0x10f6f9970>
          └ <function wrap_app_handling_exceptions at 0x10cf2b920>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
          │   │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          └ <fastapi.middleware.asyncexitstack.AsyncExitStackMiddleware object at 0x10f6f9670>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
          │    │   │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <fastapi.routing.APIRouter object at 0x10f576c90>
          └ <fastapi.middleware.asyncexitstack.AsyncExitStackMiddleware object at 0x10f6f9670>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
          │    │                │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │    │                │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │                └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <bound method Router.app of <fastapi.routing.APIRouter object at 0x10f576c90>>
          └ <fastapi.routing.APIRouter object at 0x10f576c90>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
          │     │      │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │     │      │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │     │      └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │     └ <function Route.handle at 0x10cf59120>
          └ APIRoute(path='/api/servers/{profile_id}/start', name='start_server', methods=['POST'])
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/routing.py", line 290, in handle
    await self.app(scope, receive, send)
          │    │   │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │    │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │    │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │    └ <function request_response.<locals>.app at 0x10f659e40>
          └ APIRoute(path='/api/servers/{profile_id}/start', name='start_server', methods=['POST'])
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 115, in app
    await wrap_app_handling_exceptions(app, request)(scope, receive, send)
          │                            │    │        │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b375b20>
          │                            │    │        │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │                            │    │        └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          │                            │    └ <starlette.requests.Request object at 0x129d957f0>
          │                            └ <function request_response.<locals>.app.<locals>.app at 0x11b376b60>
          └ <function wrap_app_handling_exceptions at 0x10cf2b920>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
          │   │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b376ac0>
          │   │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │   └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          └ <function request_response.<locals>.app.<locals>.app at 0x11b376b60>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/fastapi/routing.py", line 102, in app
    await response(scope, receive, send)
          │        │      │        └ <function wrap_app_handling_exceptions.<locals>.wrapped_app.<locals>.sender at 0x11b376ac0>
          │        │      └ <function RequestResponseCycle.receive at 0x11b3b5620>
          │        └ {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'}, 'http_version': '1.1', 'server': ('127.0.0.1', 10241), 'c...
          └ <starlette.responses.JSONResponse object at 0x129d95f40>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/responses.py", line 167, in __call__
    await self.background()
          │    └ <fastapi.background.BackgroundTasks object at 0x12928e390>
          └ <starlette.responses.JSONResponse object at 0x129d95f40>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/background.py", line 36, in __call__
    await task()
          └ <starlette.background.BackgroundTask object at 0x129f5af30>
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/starlette/background.py", line 21, in __call__
    await self.func(*self.args, **self.kwargs)
          │    │     │    │       │    └ {}
          │    │     │    │       └ <starlette.background.BackgroundTask object at 0x129f5af30>
          │    │     │    └ ()
          │    │     └ <starlette.background.BackgroundTask object at 0x129f5af30>
          │    └ <function start_server.<locals>.load_model at 0x111139440>
          └ <starlette.background.BackgroundTask object at 0x129f5af30>

  File "/Users/atomasini/Development/mlx-manager/backend/mlx_manager/routers/servers.py", line 318, in load_model
    await pool.get_model(model_id)
          │    │         └ 'mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit'
          │    └ <function ModelPoolManager.get_model at 0x10db813a0>
          └ <mlx_manager.mlx_server.models.pool.ModelPoolManager object at 0x10f5aaae0>

  File "/Users/atomasini/Development/mlx-manager/backend/mlx_manager/mlx_server/models/pool.py", line 289, in get_model
    return await self._load_model(model_id)
                 │    │           └ 'mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit'
                 │    └ <function ModelPoolManager._load_model at 0x10db81440>
                 └ <mlx_manager.mlx_server.models.pool.ModelPoolManager object at 0x10f5aaae0>

> File "/Users/atomasini/Development/mlx-manager/backend/mlx_manager/mlx_server/models/pool.py", line 350, in _load_model
> model = await asyncio.to_thread(load_audio, model_id)
>              │       │         │           └ 'mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit'
>              │       │         └ <function load_model at 0x129d70400>
>              │       └ <function to_thread at 0x105e26840>
>              └ <module 'asyncio' from '/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/asyncio/__i...

  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/asyncio/threads.py", line 25, in to_thread
    return await loop.run_in_executor(None, func_call)
                 │    │                     └ functools.partial(<built-in method run of _contextvars.Context object at 0x1271c4a80>, <function load_model at 0x129d70400>, ...
                 │    └ <cyfunction Loop.run_in_executor at 0x10b19a980>
                 └ <uvloop.Loop running=True closed=False debug=False>
  File "/Users/atomasini/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/lib/python3.12/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
             │        │            └ None
             │        └ None
             └ None
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/logfire/_internal/integrations/executors.py", line 53, in _run_with_context
    return func()
           └ functools.partial(<built-in method run of _contextvars.Context object at 0x1271c4a80>, <function load_model at 0x129d70400>, ...
  File "/Users/atomasini/Development/mlx-manager/backend/.venv/lib/python3.12/site-packages/mlx_audio/utils.py", line 677, in load_model
    raise ValueError(f"Could not determine model type for {model_name}")
                                                           └ 'mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit'

ValueError: Could not determine model type for mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit
