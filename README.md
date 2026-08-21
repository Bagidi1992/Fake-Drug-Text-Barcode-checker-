# Fake-Drug-Text-Barcode-checker-
A machine learning pipeline designed to flag suspicious drug registrations and fake NAFDAC Reg Nos (NRNs) based on string mutation analysis and TF-IDF feature extraction.

## Pipeline Overview
1.Data Ingestion**: Loads authentic NAFDAC product database entries.
2.Synthetic Generation**: Simulates counterfeits via leetspeak mutations and invalid NRN prefix injection.
3.Feature Engineering**: Character and word-level $n$-gram TF-IDF vectorization.
4. Classification**: Random Forest Classifier evaluating authenticity likelihood.
