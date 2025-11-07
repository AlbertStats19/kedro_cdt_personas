#!/bin/bash -ex
# *****************************
# **   Date: 29/09/2019      **
# ** Author: ADL DevOps Team **
# *****************************

echo "Waiting for Localstack to be ready..."
waitforit -address=http://localstack:4572 -timeout=120 -retry=10 -debug

echo "AWS is UP - Executing commands..."

echo "Creation test enviroment for validation"
aws --endpoint-url=http://localstack:4572 s3 mb s3://adl-landing-test
# aws --endpoint-url=http://localstack:4572 s3 cp data/DATA_EXAMPLE.csv s3://adl-landing-test/DATA_EXAMPLE.csv

exit 0
