#!/usr/bin/env bash
# Setup IAM service role for Bedrock → CloudWatch Logs invocation logging.
# Run once per AWS account. Requires IAM + Logs permissions.
#
# Usage:
#   export AWS_REGION=us-east-1
#   ./scripts/setup_bedrock_logging.sh

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ROLE_NAME="BedrockCloudWatchLogsRole"
POLICY_NAME="BedrockCloudWatchLogsPolicy"
LOG_GROUP="/aws/bedrock/model-invocation-logs"

echo "==> Region: ${REGION}"

# 1. Create CloudWatch Log Group (idempotent)
echo "==> Creating log group: ${LOG_GROUP}"
aws logs create-log-group \
  --log-group-name "${LOG_GROUP}" \
  --region "${REGION}" 2>/dev/null || echo "    (log group already exists)"

# Set retention to 30 days to control costs
aws logs put-retention-policy \
  --log-group-name "${LOG_GROUP}" \
  --retention-in-days 30 \
  --region "${REGION}"
echo "    Retention set to 30 days"

# 2. Create IAM trust policy (allows Bedrock service to assume the role)
TRUST_POLICY=$(cat <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)

# 3. Create the IAM role
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "==> Account: ${ACCOUNT_ID}"
echo "==> Creating IAM role: ${ROLE_NAME}"

aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document "${TRUST_POLICY}" \
  --description "Allows Amazon Bedrock to write model invocation logs to CloudWatch" \
  2>/dev/null || echo "    (role already exists)"

# 4. Create and attach the permissions policy
PERMISSIONS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:${LOG_GROUP}:*"
    }
  ]
}
EOF
)

echo "==> Attaching permissions policy: ${POLICY_NAME}"
aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "${POLICY_NAME}" \
  --policy-document "${PERMISSIONS_POLICY}"

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo ""
echo "==> Done!"
echo ""
echo "Role ARN: ${ROLE_ARN}"
echo "Log Group: ${LOG_GROUP}"
echo ""
echo "Next steps — enable logging in Bedrock console:"
echo "  1. Go to: https://console.aws.amazon.com/bedrock/home?region=${REGION}#/settings"
echo "  2. Under 'Model invocation logging', click Edit"
echo "  3. Turn ON logging"
echo "  4. Select 'CloudWatch Logs only'"
echo "  5. For Log group name, enter: ${LOG_GROUP}"
echo "  6. For Service role, select: ${ROLE_NAME}"
echo "  7. Click Save"
