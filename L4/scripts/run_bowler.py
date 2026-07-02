from bowler import Query
import sys

target = sys.argv[1] if len(sys.argv) > 1 else "L4/src/iron_aging_gnn/training/trainer_bowler_test.py"

Query().select_function("train_sage").rename("train_sage_v2").execute()
