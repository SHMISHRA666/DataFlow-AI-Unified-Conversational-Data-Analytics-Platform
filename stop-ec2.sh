#!/bin/bash
INSTANCE_ID=""

echo "🛑 Stopping EC2 instance..."
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "✅ Instance stopped — only storage charges now!"