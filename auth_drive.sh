#!/usr/bin/env bash
# Authenticate gcloud application-default credentials with Google Drive scope.
# Run this once when credentials expire before executing pipeline.ipynb locally.
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/cloud-platform
