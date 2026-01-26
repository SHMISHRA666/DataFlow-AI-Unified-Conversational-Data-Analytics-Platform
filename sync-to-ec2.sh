#!/bin/bash

# Configuration
INSTANCE_ID="i-0545a580337d6410c"
KEY_FILE="/Users/vibhanshuray/Downloads/EAG.pem"
IP="13.234.11.1"

echo "📦 Syncing code to EC2 using scp..."

# Create temporary archive excluding unwanted files
echo "📁 Creating archive..."
tar --exclude-from='.syncignore' -czf temp-sync.tar.gz .

# Copy archive to EC2
echo "📤 Uploading to EC2..."
scp -i "$KEY_FILE" -o StrictHostKeyChecking=no temp-sync.tar.gz ubuntu@$IP:~/

# Extract on EC2 and cleanup
echo "📂 Extracting on EC2..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ubuntu@$IP "cd ~/my-app && tar -xzf ~/temp-sync.tar.gz && rm ~/temp-sync.tar.gz"

# Cleanup local temp file
rm temp-sync.tar.gz

echo "✅ Code sync complete!"
echo "🔗 Connect: ssh -i \"$KEY_FILE\" ubuntu@$IP"