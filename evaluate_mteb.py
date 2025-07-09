#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import mteb
from mteb import MTEB
from mteb.model_meta import ModelMeta # Import ModelMeta

# Import your custom models and their ModelMeta objects
# This import assumes mteb_custom_models.py is directly in mteb/models/ or
# accessible via PYTHONPATH. For a real MTEB PR, it would be in mteb/models/.
from mteb_custom_models import olmo_7b_base, openelm_3b_base

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on MTEB benchmark tasks.")
    
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate (e.g., "openelm", "olmo")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding') # Increased default batch size
    parser.add_argument('--output_dir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--device', type=str, default="auto", help='Device for model (e.g., "cuda", "cpu", "auto")')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use ModelMeta to get the model loader and instantiate the model
    if args.model_name.lower() in ["openelm", "openelm-3b-base-mteb"]:
        # The loader is a callable that returns an Encoder instance
        model_meta: ModelMeta = openelm_3b_base
    elif args.model_name.lower() in ["olmo", "olmo-7b-base-mteb"]:
        model_meta: ModelMeta = olmo_7b_base
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}. "
                         f"Please choose 'openelm' or 'olmo'.")

    # Instantiate the model using the loader from ModelMeta
    # Pass common args like device and cache_dir if your loader accepts them.
    # Note: If your models have fixed cache_dir in their loader, remove cache_dir from here.
    model = model_meta.loader(device=args.device) 
    
    # desired_task_types = [
    #     "Clustering",
    #     "PairClassification",
    #     "Reranking",
    #     "Retrieval",
    #     "STS",
    #     "Summarization"
    # ]
    desired_task_types = [
        # "Summarization",
        # "STS",
        "Retrieval",
        # "Reranking",
        # "PairClassification",
        # "Clustering"
    ]
    
    mteb_pair_classification_tasks = [
    "LegalBenchPC",
    "PubChemAISentenceParaphrasePC",
    # "PubChemSMILESPC", # not working
    "PubChemSynonymPC",
    "PubChemWikiParagraphsPC",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
    "OpusparcusPC",
    "PawsXPairClassification",
    "PubChemWikiPairClassification",
    "RTE3",
    "XNLI",
]
    
    retrieval_datasets = [
    "CQADupstackRetrieval",
    "AppsRetrieval",
    "CodeFeedbackMT",
    "CodeFeedbackST",
    "CosQA",
    "StackOverflowQA",
    "SyntheticText2SQL",
    "AILACasedocs",
    "AILAStatutes",
    "AlphaNLI",
    "ARCChallenge",
    "ArguAna",
    "BrightRetrieval", # very big dataset. 
    "BrightLongRetrieval",
    "BuiltBenchRetrieval",
    "ChemHotpotQARetrieval",
    "ChemNQRetrieval",
    "ClimateFEVER",
    "ClimateFEVERHardNegatives",
    "ClimateFEVER.v2",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "DBPediaHardNegatives",
    "FaithDial",
    "FeedbackQARetrieval",
    "FEVER",
    "FEVERHardNegatives",
    "FiQA2018",
    "HagridRetrieval",
    "HellaSwag",
    "HotpotQA",
    "HotpotQAHardNegatives",
    "LegalBenchConsumerContractsQA",
    "LegalBenchCorporateLobbying",
    "LegalSummarization",
    "LEMBNarrativeQARetrieval",
    "LEMBNeedleRetrieval",
    "LEMBPasskeyRetrieval",
    "LEMBQMSumRetrieval",
    "LEMBSummScreenFDRetrieval",
    "LEMBWikimQARetrieval",
    "LitSearchRetrieval",
    "MedicalQARetrieval",
    "MLQuestions",
    "MSMARCO",
    "MSMARCOHardNegatives",
    "MSMARCOv2",
    "NanoArguAnaRetrieval",
    "NanoClimateFeverRetrieval",
    "NanoDBPediaRetrieval",
    "NanoFEVERRetrieval",
    "NanoFiQA2018Retrieval",
    "NanoHotpotQARetrieval",
    "NanoMSMARCORetrieval",
    "NanoNFCorpusRetrieval",
    "NanoNQRetrieval",
    "NanoQuoraRetrieval",
    "NanoSCIDOCSRetrieval",
    "NanoSciFactRetrieval",
    "NanoTouche2020Retrieval",
    "NarrativeQARetrieval",
    "NFCorpus",
    "NQ",
    "NQHardNegatives",
    "PIQA",
    "Quail",
    "QuoraRetrieval",
    "QuoraRetrievalHardNegatives",
    "R2MEDBiologyRetrieval",
    "R2MEDBioinformaticsRetrieval",
    "R2MEDMedicalSciencesRetrieval",
    "R2MEDMedXpertQAExamRetrieval",
    "R2MEDMedQADiagRetrieval",
    "R2MEDPMCTreatmentRetrieval",
    "R2MEDPMCClinicalRetrieval",
    "R2MEDIIYiClinicalRetrieval",
    "RARbCode",
    "RARbMath",
    "SCIDOCS",
    "SciFact",
    "SIQA",
    "SpartQA",
    "TempReasonL1",
    "TempReasonL2Context",
    "TempReasonL2Fact",
    "TempReasonL2Pure",
    "TempReasonL3Context",
    "TempReasonL3Fact",
    "TempReasonL3Pure",
    "TopiOCQA",
    "TopiOCQAHardNegatives",
    "Touche2020Retrieval.v3",
    "TRECCOVID",
    "WinoGrande",
    "BelebeleRetrieval",
    "CUREv1",
    "MIRACLRetrieval",
    "MIRACLRetrievalHardNegatives",
    "MLQARetrieval",
    "MrTidyRetrieval",
    "MultiLongDocRetrieval",
    "PublicHealthQA",
    "StatcanDialogueDatasetRetrieval",
    "WebFAQRetrieval",
    "WikipediaRetrievalMultilingual",
    "XMarket",
    "XPQARetrieval",
    "XQuADRetrieval",
    "VDRMultilingualRetrieval",
    "mFollowIRCrossLingualInstructionRetrieval",
]

    
    print(f"\nFetching MTEB tasks for evaluation (English only, excluding bitext mining):")
    print(f"  Task types: {desired_task_types}")
    
    
    # For initial testing, you can limit to a smaller set of tasks or a single task
    tasks = mteb.get_tasks(task_types=desired_task_types, languages=["eng"]) # Example single task
    # tasks = mteb.get_tasks(tasks=mteb_pair_classification_tasks, languages=["eng"]) # Example single task


    evaluation = MTEB(tasks=tasks)
    
    encode_kwargs = {"batch_size": args.batch_size}
    print("\nRunning MTEB evaluation...")
    results = evaluation.run(
        model,
        verbosity=3, # Reduced verbosity for cleaner output, can be 3 for more details
        encode_kwargs=encode_kwargs,
        output_folder=str(output_dir / model_meta.name.replace('/', '_')) # Use model_meta.name
    )
    print("results:", results)

    # Print and save results
    print("\n--- MTEB Evaluation Results ---")
    try:
        for task_result_dict in results:
            # Each item in `results` is a dictionary for a single task.
            # It typically has keys like 'task_name', 'scores', 'main_score', etc.
            task_name = task_result_dict.get('task_name', 'Unknown Task')
            main_score = task_result_dict.get('main_score', 'N/A')
            
            print(f"\nTask: {task_name} (Main Score: {main_score:.4f} if number else {main_score})")
            
            # Access the detailed scores, if available
            scores_detail = task_result_dict.scores
            if scores_detail:
                # Scores might be nested (e.g., test split scores)
                for split_name, split_metrics in scores_detail.items():
                    print(f"  Split: {split_name}")
                    for metric_name, metric_value in split_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            print(f"    {metric_name}: {metric_value:.4f}")
                        elif isinstance(metric_value, dict):
                            print(f"    {metric_name}:")
                            for sub_metric_name, sub_metric_value in metric_value.items():
                                print(f"      {sub_metric_name}: {sub_metric_value:.4f}")
                        else:
                            print(f"    {metric_name}: {metric_value}")
            else:
                print("  No detailed scores available for this task.")
        # Save to JSON
        result_path = output_dir / f"mteb_results_{model_meta.name.replace('/', '_')}.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_path}")

    except Exception as e:
        print("results:", results)
        print(f"\nAn error occurred during MTEB evaluation: {e}")
        print("Common issues: ")
        print("- Missing datasets (MTEB downloads them, requires internet)")
        print("- Insufficient memory (some models/tasks are resource-intensive)")
        print("- Incorrect task names/types")
        print("- Ensure the specified 'target_gpu_device' is actually available.")
    

if __name__ == "__main__":
    main()