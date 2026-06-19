import logging
from app.rag.rag_agent import agent
from app.rag.rag_reports import generate_safety_report, get_worker_risk_profile
from app.rag.rag_evaluation import evaluate_rag_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("PPE Safetly SQL Assistant")
    print("\n1. Ask Question")
    print("2. Daily Report")
    print("3. Worker risk profile")
    print("4. Evaluate RAG quality")
    print("5. Quit")

    while True:
        try:
            choice = input("\nChoose (1-5): ").strip()

            if choice == "1":
                question = input("Ask: ").strip()
                if not question:
                    print("Please enter a question")
                    continue
                result = agent.invoke({"input": question})
                output = result["output"]
                if isinstance(output, list):
                    output = " ".join([
                        item["text"] for item in output
                        if isinstance(item, dict) and "text" in item
                    ])
                print(f"\nAnswer: {output}\n")
            
            elif choice == "2":
                print("\nGenerating report...")
                report = generate_safety_report()
                print(f"\nDaily Report")
                print(f"Date: {report.report_date}")
                print(f"Total Violations: {report.total_violations}")
                print(f"Hardhat: {report.hardhat_violations}")
                print(f"Vest: {report.vest_violations}")
                print(f"Severity: {report.severity}")
                print(f"Compliance Rate: {report.compliance_rate:.1f}%")
                print(f"Recommendation: {report.recommendation}")
            
            elif choice == "3":
                pid = input("Person ID: ").strip()
                if not pid.isdigit():
                    print("Please enter a valid number")
                    continue
                profile = get_worker_risk_profile(int(pid))
                print(f"\nWorker {pid} Risk Profile")
                print(f"Total Violations: {profile.total_violations}")
                print(f"Hardhat: {profile.hardhat_violations}")
                print(f"Vest: {profile.vest_violations}")
                print(f"Risk Level: {profile.risk_level}")
                print(f"Action Required: {profile.action_required}")
            
            elif choice == "4":
                print("\nRunning RAGAS evaluation...")
                scores = evaluate_rag_quality()
                if scores:
                    print("\nRAGAS Results")
                    print(f"Faithfulness: {scores['faithfulness']:.3f}")
                    print(f"Answer Relevancy: {scores['answer_relevancy']:.3f}")
            
            elif choice == "5":
                break

            else:
                print("Invalid choice — enter 1-5")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()