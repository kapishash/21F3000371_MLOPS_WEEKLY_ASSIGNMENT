import json

INPUT_FILE = "data/week10/iris_v2_description.jsonl"
OUTPUT_FILE = "data/week10/iris_v2_gemini.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    count = 0

    for line in infile:
        if not line.strip():
            continue

        record = json.loads(line)

        converted = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": record["input_text"]
                        }
                    ]
                },
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": record["output_text"]
                        }
                    ]
                }
            ]
        }

        outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
        count += 1

print(f"Converted {count} records")
print(f"Output: {OUTPUT_FILE}")