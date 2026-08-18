import json

INPUT_FILE = "data/week10/iris_v1_raw.jsonl"
OUTPUT_FILE = "data/week10/iris_v1_gemini.jsonl"

with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for line in infile:
        row = json.loads(line)

        example = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": row["input_text"]
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": row["output_text"]
                        }
                    ]
                }
            ]
        }

        outfile.write(json.dumps(example) + "\n")

print("Converted dataset successfully.")
print("Output:", OUTPUT_FILE)
