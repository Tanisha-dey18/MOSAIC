---
title: MOSAIC API
emoji: 🧩
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Backend API for the MOSAIC wearable anomaly research project
---

# MOSAIC API

FastAPI backend of [MOSAIC](https://github.com/Tanisha-dey18/MOSAIC), a research platform for personalised anomaly detection in wearable physiological signals. The frontend is hosted on Netlify.

On start-up the server downloads the open PhysioNet *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (v1.0.1, ODC-By 1.0, DOI 10.13026/he0v-tf17), verifies its checksums, processes it and trains the models. This takes a few minutes; progress is visible in the app.

Research software only. It is not a medical device and is not for diagnosis.

API documentation: `/docs`. Health check: `/api/health`.
