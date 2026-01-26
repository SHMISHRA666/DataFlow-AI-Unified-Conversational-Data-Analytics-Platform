#!/bin/bash
INSTANCE_ID="i-0545a580337d6410c"

echo "🚀 Starting EC2 instance..."
aws ec2 start-instances --instance-ids $INSTANCE_ID
echo "⏳ Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
echo "✅ Instance started!"