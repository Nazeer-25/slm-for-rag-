# Free hosting

The live website runs real inference with Ollama on your own computer, so it cannot be hosted for free in the cloud reliably. There are three free options, depending on what you need.

| Option | Live answers | Who can open it | Cost |
|---|---|---|---|
| A. Static snapshot on GitHub Pages | No (recorded runs) | Anyone, permanently | Free |
| B. Same Wi-Fi / LAN | Yes, from your PC | People on your network | Free |
| C. Temporary public link (Cloudflare quick tunnel) | Yes, from your PC | Anyone with the link, while your PC runs | Free |

## A. Static snapshot on GitHub Pages (recommended for sharing)

The snapshot shows the datasets, architecture, saved detector and benchmark results, the six scenarios and the recorded runs. It cannot generate new answers.

1. Run the scenarios you want to show on the live website (they are recorded in the history, up to 30 runs). Generate the benchmark too if you want its charts included.
2. Export into the `docs/showcase` folder:
   ```bat
   .venv\Scripts\python.exe scripts\export_showcase.py --output docs\showcase
   ```
3. Commit and push it:
   ```bat
   git add docs\showcase
   git commit -m "Add static showcase"
   git push
   ```
4. On GitHub, open the repository → **Settings → Pages**. Under *Build and deployment*, choose **Deploy from a branch**, branch `main`, folder `/docs`, then **Save**.
5. After a minute or two the snapshot is at `https://<your-user>.github.io/<repository>/showcase/`. For this repository: https://nazeer-25.github.io/slm-for-rag-/showcase/

Re-run steps 2-3 whenever you want to publish newer runs. The snapshot also works offline: double-click `index.html` in the exported folder.

GitHub Pages is free for public repositories. Only publish runs you are happy to make public; custom poison text you typed is included in the recorded traces.

## B. Same Wi-Fi or LAN (for a room)

Start the website on all network interfaces:

```bat
set WORKBENCH_HOST=0.0.0.0
start_demo.bat
```

Allow Python through the Windows firewall when Windows asks (private networks only). Find your IP address with `ipconfig` (IPv4 address, for example `192.168.1.23`) and share `http://192.168.1.23:8000`. Everyone shares one workspace and one history, and every query runs on your PC.

## C. Temporary public link

Cloudflare's quick tunnels give a random public `https://….trycloudflare.com` address for a server on your PC, without an account.

1. Install `cloudflared` (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/), for example `winget install --id Cloudflare.cloudflared`.
2. Start the website normally (`start_demo.bat`), then in a second terminal:
   ```bat
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Share the printed `trycloudflare.com` link. Close the second terminal to take it offline.

The website has no login. Anyone with the link can run queries on your computer, read the history and clear it, so share the link only for the duration of a demo.

## Why not free cloud hosting for live inference?

Live answers need a machine that keeps a language model in memory. Free cloud tiers either sleep, have too little memory, or are too slow on CPU for llama3.2:3b. The remote provider (`LLM_PROVIDER=openai`) removes the need for Ollama but is not free. For a permanent public link, use option A; for a live demo, run it on your PC (options B or C).
