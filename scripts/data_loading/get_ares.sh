#!/bin/bash

URLS=("http://sensembert.org/ares_embedding.tar.gz")
FILES=("ares_embedding.tar.gz")

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
    fi
done

tar xvf ares_embedding.tar.gz
rm ares_embedding.tar.gz
rm ares_embedding/ares_bert_base_multilingual.txt
rm ares_embedding/LICENSE.txt
rm ares_embedding/README

cd ..
