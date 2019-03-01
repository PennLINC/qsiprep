#!/bin/bash

MNI_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/580705eb594d9001ed622649"
MNI_SHA256="608b1d609255424d51300e189feacd5ec74b04e244628303e802a6c0b0f9d9db"
ASYM_09C_TEMPLATE_OLD="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/580705089ad5a101f17944a9"
ASYM_09C_SHA256_OLD="a24699ba0d13f72d0f8934cc211cb80bfd9c9a077b481d9b64295cf5275235a9"
ASYM_09C_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/5b0dbce20f461a000db8fa3d"
ASYM_09C_SHA256="2851302474359c2c48995155aadb48b861e5dcf87aefda71af8010f671e8ed66"
OASIS_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/584123a29ad5a1020913609d"
OASIS_SHA256="d87300e91346c16f55baf6f54f5f990bc020b61e8d5df9bcc3abb0cc4b943113"
NKI_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/5bc3fad82aa873001bc5a553"
NKI_SHA256="9c08713d067bcf13baa61b01a9495e526b55d1f148d951da01e082679f076fa9"
OASIS_DKT31_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/5b16f17aeca4a80012bd7542"
OASIS_DKT31_SHA256="623fa7141712b1a7263331dba16eb069a4443e9640f52556c89d461611478145"
EPI_TEMPLATE="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/5bc12155ac011000176bff82"
EPI_SHA256="fcd6980ef98c9d7622c6dc2a7747ff51ba3909d98e2a740df9a8265d50920d1b"
OASIS30="https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/5b0dbce34c28ef0012c7f788"
OASIS30_SHA256="b7202abbca2c69b514a68b8457e3f718a57ccac2c2990fcf7f27ab12f1698645"

GET(){
    URL=$1; SHA256=$2;
    mkfifo pipe.tar.gz
    cat pipe.tar.gz | tar zxv -C $CRN_SHARED_DATA & SHASUM=$(curl -sSL $URL | tee pipe.tar.gz | sha256sum | cut -d\  -f 1)
    rm pipe.tar.gz

    if [[ "$SHASUM" != "$SHA256" ]]; then
        echo "Failed checksum!"
        return 1
    fi
    return 0
}

set -e
echo "Getting MNI template"
GET "$MNI_TEMPLATE" "$MNI_SHA256"
echo "Getting (deprecated version of) MNI152NLin2009cAsym template"
GET "$ASYM_09C_TEMPLATE_OLD" "$ASYM_09C_SHA256_OLD"
echo "Getting MNI152NLin2009cAsym template"
GET "$ASYM_09C_TEMPLATE" "$ASYM_09C_SHA256"
echo "Getting OASIS template"
GET "$OASIS_TEMPLATE" "$OASIS_SHA256"
echo "Getting NKI template"
GET "$NKI_TEMPLATE" "$NKI_SHA256"
echo "Getting OASIS DKT31 template"
GET "$OASIS_DKT31_TEMPLATE" "$OASIS_DKT31_SHA256"
echo "Getting qsiprep's BOLDref template"
GET "$EPI_TEMPLATE" "$EPI_SHA256"
echo "Getting OASIS30ANTs"
GET "$OASIS30" "$OASIS30_SHA256"
echo "Done!"
