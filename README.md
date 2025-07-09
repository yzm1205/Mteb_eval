# MTEB LLM Evaluation Script

This script evaluates HuggingFace LLMs (e.g., OpenELM, OLMo) on the MTEB benchmark for selected English tasks.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python evaluate_mteb.py \
  --model apple/OpenELM-3B \
  --batch_size 2 \
  --output_dir results \
  --device auto
```

- `--model`: HuggingFace model name or local path
- `--batch_size`: Batch size for encoding (default: 1)
- `--output_dir`: Directory to save results (default: `results`)
- `--device`: Device for model (e.g., `cuda`, `cpu`, `auto`)

## Output

- Results are printed to the console and saved as a JSON file in the output directory.

## Tasks Evaluated
- Clustering                                                           
    - BuiltBenchClusteringP2P, p2p                                      
    - BuiltBenchClusteringS2S, s2s                                      
    - WikiCitiesClustering, p2p                                         
    - WikipediaSpecialtiesInChemistryClustering, s2p                    
    - WikipediaChemistryTopicsClustering, s2p                          
    - MasakhaNEWSClusteringP2P, p2p, multilingual 1 / 16 Subsets       
    - MasakhaNEWSClusteringS2S, s2s, multilingual 1 / 16 Subsets       
    - ArXivHierarchicalClusteringP2P, p2p                              
    - ArXivHierarchicalClusteringS2S, p2p                              
    - BigPatentClustering.v2, p2p                                      
    - BiorxivClusteringP2P.v2, p2p                                     
    - BiorxivClusteringS2S.v2, s2s                                     
    - ClusTREC-Covid, p2p, multilingual 2 / 2 Subsets                  
    - MedrxivClusteringP2P.v2, p2p                                     
    - MedrxivClusteringS2S.v2, s2s                                     
    - RedditClustering.v2, s2s                                         
    - RedditClusteringP2P.v2, p2p                                      
    - StackExchangeClustering.v2, s2s                                  
    - StackExchangeClusteringP2P.v2, p2p                               
    - TwentyNewsgroupsClustering.v2, s2s                               
    - SIB200ClusteringS2S, s2s, multilingual 1 / 197 Subsets  

- PairClassification
    - LegalBenchPC, s2s                                                            
    - PubChemAISentenceParaphrasePC, s2s                                           
    - PubChemSMILESPC, s2s                                                         
    - PubChemSynonymPC, s2s                                                        
    - PubChemWikiParagraphsPC, p2p                                                
    - SprintDuplicateQuestions, s2s
    - TwitterSemEval2015, s2s
    - TwitterURLCorpus, s2s
    - OpusparcusPC, s2s, multilingual 1 / 6 Subsets
    - PawsXPairClassification, s2s, multilingual 1 / 7 Subsets
    - PubChemWikiPairClassification, s2s, multilingual 12 / 12 Subsets
    - RTE3, s2s, multilingual 1 / 4 Subsets
    - XNLI, s2s, multilingual 1 / 14 Subsets

- Reranking
    - AskUbuntuDupQuestions, s2s                                                  
    - BuiltBenchReranking, p2p                                                    
    - MindSmallReranking, s2s
    - SciDocsRR, s2s
    - StackOverflowDupQuestions, s2s
    - WebLINXCandidatesReranking, p2p
    - ESCIReranking, s2p, multilingual 1 / 3 Subsets
    - MIRACLReranking, s2s, multilingual 1 / 18 Subsets
    - WikipediaRerankingMultilingual, s2p, multilingual 1 / 16 Subsets

- Retrieval
    - CQADupstackRetrieval, None                                                   
    - AppsRetrieval, p2p                                                           
    - CodeFeedbackMT, p2p                                                          
    - CodeFeedbackST, p2p                                                          
    - CosQA, p2p                                                                   
    - StackOverflowQA, p2p                                                         
    - SyntheticText2SQL, p2p                                                       
    - AILACasedocs, p2p                                                           
    - AILAStatutes, p2p
    - AlphaNLI, s2s
    - ARCChallenge, s2s
    - ArguAna, s2p
    - BrightRetrieval, s2p, multilingual 12 / 12 Subsets
    - BrightLongRetrieval, s2p, multilingual 8 / 8 Subsets
    - BuiltBenchRetrieval, p2p
    - ChemHotpotQARetrieval, s2p
    - ChemNQRetrieval, s2p
    - ClimateFEVER, s2p
    - ClimateFEVERHardNegatives, s2p
    - ClimateFEVER.v2, s2p
    - CQADupstackAndroidRetrieval, s2p
    - CQADupstackEnglishRetrieval, s2p
    - CQADupstackGamingRetrieval, s2p
    - CQADupstackGisRetrieval, s2p
    - CQADupstackMathematicaRetrieval, s2p
    - CQADupstackPhysicsRetrieval, s2p
    - CQADupstackProgrammersRetrieval, s2p
    - CQADupstackStatsRetrieval, s2p
    - CQADupstackTexRetrieval, s2p                                                 
    - CQADupstackUnixRetrieval, s2p                                                
    - CQADupstackWebmastersRetrieval, s2p                                         
    - CQADupstackWordpressRetrieval, s2p                                          
    - DBPedia, s2p
    - DBPediaHardNegatives, s2p
    - FaithDial, s2p
    - FeedbackQARetrieval, s2p
    - FEVER, s2p
    - FEVERHardNegatives, s2p
    - FiQA2018, s2p
    - HagridRetrieval, s2p
    - HellaSwag, s2s
    - HotpotQA, s2p
    - HotpotQAHardNegatives, s2p
    - LegalBenchConsumerContractsQA, s2p
    - LegalBenchCorporateLobbying, s2p
    - LegalSummarization, s2p
    - LEMBNarrativeQARetrieval, s2p
    - LEMBNeedleRetrieval, s2p
    - LEMBPasskeyRetrieval, s2p
    - LEMBQMSumRetrieval, s2p
    - LEMBSummScreenFDRetrieval, s2p
    - LEMBWikimQARetrieval, s2p                                                    
    - LitSearchRetrieval, s2p                                                     
    - MedicalQARetrieval, s2s                                                     
    - MLQuestions, s2p                                                            
    - MSMARCO, s2p                                                                
    - MSMARCOHardNegatives, s2p
    - MSMARCOv2, s2p
    - NanoArguAnaRetrieval, s2p
    - NanoClimateFeverRetrieval, s2p
    - NanoDBPediaRetrieval, s2p
    - NanoFEVERRetrieval, s2p
    - NanoFiQA2018Retrieval, s2p
    - NanoHotpotQARetrieval, s2p
    - NanoMSMARCORetrieval, s2p
    - NanoNFCorpusRetrieval, s2p
    - NanoNQRetrieval, s2p
    - NanoQuoraRetrieval, s2s
    - NanoSCIDOCSRetrieval, s2p
    - NanoSciFactRetrieval, s2p
    - NanoTouche2020Retrieval, s2p
    - NarrativeQARetrieval, s2p
    - NFCorpus, s2p
    - NQ, s2p                                                                     
    - NQHardNegatives, s2p                                                        
    - PIQA, s2s                                                                   
    - Quail, s2s                                                                  
    - QuoraRetrieval, s2s                                                         
    - QuoraRetrievalHardNegatives, s2s                                            
    - R2MEDBiologyRetrieval, s2p                                                  
    - R2MEDBioinformaticsRetrieval, s2p                                           
    - R2MEDMedicalSciencesRetrieval, s2p                                          
    - R2MEDMedXpertQAExamRetrieval, s2p                                           
    - R2MEDMedQADiagRetrieval, s2p                                                
    - R2MEDPMCTreatmentRetrieval, s2p
    - R2MEDPMCClinicalRetrieval, s2p
    - R2MEDIIYiClinicalRetrieval, s2p
    - RARbCode, s2p
    - RARbMath, s2p
    - SCIDOCS, s2p
    - SciFact, s2p
    - SIQA, s2s
    - SpartQA, s2s
    - TempReasonL1, s2s
    - TempReasonL2Context, s2s                                                     
    - TempReasonL2Fact, s2s                                                        
    - TempReasonL2Pure, s2s                                                        
    - TempReasonL3Context, s2s                                                     
    - TempReasonL3Fact, s2s                                                        
    - TempReasonL3Pure, s2s                                                        
    - TopiOCQA, s2p                                                               
    - TopiOCQAHardNegatives, s2p
    - Touche2020Retrieval.v3, s2p
    - TRECCOVID, s2p
    - WinoGrande, s2s
    - BelebeleRetrieval, s2p, multilingual 243 / 376 Subsets
    - CUREv1, s2p, multilingual 3 / 3 Subsets
    - MIRACLRetrieval, s2p, multilingual 1 / 18 Subsets
    - MIRACLRetrievalHardNegatives, s2p, multilingual 1 / 18 Subsets
    - MLQARetrieval, s2p, multilingual 13 / 49 Subsets
    - MrTidyRetrieval, s2p, multilingual 1 / 11 Subsets
    - MultiLongDocRetrieval, s2p, multilingual 1 / 13 Subsets
    - PublicHealthQA, s2p, multilingual 1 / 8 Subsets
    - StatcanDialogueDatasetRetrieval, s2p, multilingual 1 / 2 Subsets
    - WebFAQRetrieval, s2p, multilingual 1 / 49 Subsets
    - WikipediaRetrievalMultilingual, s2p, multilingual 1 / 16 Subsets
    - XMarket, s2p, multilingual 1 / 3 Subsets
    - XPQARetrieval, s2p, multilingual 24 / 36 Subsets
    - XQuADRetrieval, s2p, multilingual 1 / 12 Subsets
    - VDRMultilingualRetrieval, it2it, multilingual 1 / 5 Subsets
    - mFollowIRCrossLingualInstructionRetrieval, s2p, multilingual 3 / 3 Subsets

- STS
    - BIOSSES, s2s                                                                
    - SICK-R, s2s                                                                 
    - STS12, s2s                                                                  
    - STS13, s2s                                                                  
    - STS14, s2s                                                                  
    - STS15, s2s                                                                  
    - STS16, s2s                                                                  
    - STSBenchmark, s2s                                                           
    - IndicCrosslingualSTS, s2s, multilingual 12 / 12 Subsets                     
    - SemRel24STS, s2s, multilingual 1 / 12 Subsets                               
    - STS17, s2s, multilingual 8 / 11 Subsets                                     
    - STS22.v2, p2p, multilingual 5 / 18 Subsets                                  
    - STSBenchmarkMultilingualSTS, s2s, multilingual 1 / 10 Subsets
- Summarization
     - SummEvalSummarization.v2, p2p

## Reference
- [MTEB GitHub](https://github.com/embeddings-benchmark/mteb/tree/main) 