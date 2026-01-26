#!/bin/bash
INSTANCE_ID="i-0545a580337d6410c"

echo "🛑 Stopping EC2 instance..."
aws ec2 stop-instances --instance-ids $INSTANCE_ID
echo "✅ Instance stopped — only storage charges now!"