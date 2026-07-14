# Image Inference Server

Runs the image model (FLUX.2 Klein / FLUX.1 Kontext) as a standalone HTTP service
that loads the model **once**, keeps it warm on the GPU, and serves generation on
request. The game becomes a thin client, so it holds no model and no GPU — and any
device on the same network can use the same endpoint.

## Why run it this way

- **Warm model.** No ~50 s reload per generation; the model stays resident.
- **Crash isolation.** If the GPU path crashes, the server restarts on its own
  without taking the game down.
- **Shared + remote.** Your game and, say, your laptop on the same wifi can both
  call it.

## Running the server

On the GPU machine, set the backend to host and start it:

```bash
# In the server's environment (.env), host the fast local backend:
#   IMAGE_PROVIDER=flux-klein
uv run python image_server.py
```

Startup loads and quantizes the model, then runs one warmup generation. When you
see `✓ Warmup complete — server ready`, it's serving. Check it:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ready","backend":"FluxKleinImageGenerator","provider":"flux-klein"}
```

> The server is single-worker by design. **Do not** run it with multiple uvicorn
> workers — each would load its own copy of the model and fight over the GPU.
> Requests are serialized automatically (one generation at a time); concurrent
> callers queue.

## Pointing the game at it

In the **game's** environment (`.env`), switch to the HTTP client backend:

```ini
IMAGE_PROVIDER=http
IMAGE_SERVER_URL=http://127.0.0.1:8000     # same machine
```

Nothing else in the game changes — the client implements the same interface.

## Using it from another device on the same wifi (no static IP needed)

You do **not** need a paid static IP — that's a *public* internet address and is
irrelevant here. You only need the server's *local* (LAN) address, which is free.
Three ways, easiest first:

1. **By computer name (recommended — survives IP changes).**
   This machine is `BAXXPC`, so from your laptop just use:
   ```ini
   IMAGE_SERVER_URL=http://BAXXPC:8000
   # or, if plain name doesn't resolve, try mDNS:
   IMAGE_SERVER_URL=http://BAXXPC.local:8000
   ```
   The name works regardless of what IP the router hands out.

2. **By current LAN IP.** Right now it's `10.0.0.119`:
   ```ini
   IMAGE_SERVER_URL=http://10.0.0.119:8000
   ```
   Find it anytime with `ipconfig` (look at "IPv4 Address" for your wifi/ethernet
   adapter). This IP can change after a reboot unless you do #3.

3. **Reserve the IP on your router (free).** In your router admin page, under
   DHCP reservations, bind this machine's MAC address to a fixed LAN IP (e.g.
   `10.0.0.119`). Then the IP never changes. No ISP plan or payment involved.

### Two requirements for LAN access

- **Bind to all interfaces.** The default `IMAGE_SERVER_HOST=0.0.0.0` already does
  this. (`127.0.0.1` would be this-machine-only.)
- **Open the Windows Firewall port** on the server machine, or the laptop's
  requests will silently time out. In an **admin** PowerShell:
  ```powershell
  New-NetFirewallRule -DisplayName "Image Server 8000" -Direction Inbound `
    -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
  ```
  (`-Profile Private` keeps it to home/private networks, not public ones.)

## Security

Fine to expose on a **trusted** home LAN. Do **not** port-forward it to the public
internet — it has no rate limiting and a GPU behind it. For a little protection
(shared/office wifi), set a shared secret on **both** server and client:

```ini
IMAGE_SERVER_API_KEY=some-long-random-string
```

The server then requires header `X-API-Key: <value>`; the client sends it
automatically.

## Endpoints

| Method | Path                   | Body                                  | Returns   |
|--------|------------------------|---------------------------------------|-----------|
| GET    | `/health`              | —                                     | JSON      |
| POST   | `/generate/character`  | form: `prompt`                        | image/png |
| POST   | `/generate/scene`      | form: `prompt`, files: `references[]` | image/png |

Interactive docs at `http://<server>:8000/docs`.

## Configuration reference (`core/config.py`)

| Setting                 | Default                 | Purpose                                  |
|-------------------------|-------------------------|------------------------------------------|
| `IMAGE_SERVER_HOST`     | `0.0.0.0`               | Server bind address                      |
| `IMAGE_SERVER_PORT`     | `8000`                  | Server port                              |
| `IMAGE_SERVER_API_KEY`  | `""`                    | Optional shared secret (both sides)      |
| `IMAGE_SERVER_URL`      | `http://127.0.0.1:8000` | Client: where the server is              |
| `IMAGE_SERVER_TIMEOUT`  | `300.0`                 | Client: per-request timeout (seconds)    |
