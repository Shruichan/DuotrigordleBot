# Publishing to the Chrome Web Store

The extension package is built and ready. I can't upload it for you (the
Chrome Web Store needs *your* developer account + the one-time $5 fee +
a manual upload through the dashboard), but everything you need is here.

## What's ready

- `dist/duotrigordle-bot.zip` — the upload bundle. Has icons (16/48/128),
  manifest with author + homepage, the bundled engine, value net, and data.
- `dist/store-assets/promo-tile-440x280.png` — the small store-tile image.
- `extension/icons/icon-128.png` — also used in the store header.

## One-time setup (5 minutes)

1. Go to <https://chrome.google.com/webstore/devconsole/> and sign in with the
   Google account you want the extension published under.
2. Pay the **one-time $5 developer registration fee**. (Google requires it
   once per account; you keep the account forever.)
3. Click **New item** in the developer dashboard.
4. Upload `dist/duotrigordle-bot.zip`.

## Filling in the store listing

Paste these where the dashboard asks. Everything matches the live site +
manifest so the listing reads consistently.

- **Name:** Duotrigordle Bot
- **Short description (132 chars):**
  > A near-optimal solver for duotrigordle.com — overlays the next guess on every daily and practice game. Runs in your browser.
- **Detailed description:** (paste the section below)

```
Duotrigordle Bot plays the daily 32-Wordle on duotrigordle.com. It scores
every five-letter word against all thirty-two boards at once and overlays
the suggested guess on the page.

How well it plays
- Averages 33.6 guesses on the daily (well under the 37-guess limit).
- Caps the worst game at 35 guesses — never blows past.
- 100% solve rate over a 2,000-game benchmark on the daily distribution.

Strategy
- Entropy across all 32 boards plus a learned value-net tie-break that
  flattens the worst-case tail.
- All compute runs in your browser; no servers, no accounts, no tracking.

Open source: https://github.com/Shruichan/DuotrigordleBot
Live demo + writeup: https://shruichan.github.io/DuotrigordleBot/
Not affiliated with duotrigordle.com.
```

- **Category:** *Productivity* (or *Fun*)
- **Language:** English
- **Homepage URL:** `https://shruichan.github.io/DuotrigordleBot/`
- **Support URL:** `https://github.com/Shruichan/DuotrigordleBot/issues`

## Images

- **Store icon (128×128):** `extension/icons/icon-128.png`
- **Small promo tile (440×280):** `dist/store-assets/promo-tile-440x280.png`
- **Screenshots (1280×800 or 640×400):** none generated — take one of the
  overlay running on duotrigordle.com (a real game with the overlay panel
  visible reads best). Need at least one to publish.

## Privacy + permissions

- **Single purpose:** "Suggests guesses for the puzzle game at
  duotrigordle.com." (matches description)
- **Permission justifications:**
  - `storage` — saves the user's UI preferences locally.
  - `host_permissions: https://duotrigordle.com/*` — reads the on-page board
    state to compute suggestions; required for the overlay.
- **Data collection:** *No data collected.* Tick all "I do not collect" boxes.
  The extension runs entirely in the user's browser and makes no network
  requests beyond fetching its own bundled files.
- **Remote code:** *No.* Everything ships inside the zip.

## Submitting

Click **Submit for review** when the listing is filled in. Google takes
anywhere from a few hours to a few days; you'll get an email when it's live.

## Updating later

Just bump `version` in `extension/manifest.json`, run `scripts/package-extension.sh`
again to produce a new zip, and upload it from the dashboard's *Package* tab.
