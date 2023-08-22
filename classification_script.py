import os
from typing import List, Dict
from pydantic import BaseModel
import openai
import pandas as pd
from collections import defaultdict
import json
import tiktoken
from dotenv import load_dotenv
import logging

# Initialize the tokenizer

logger = logging.getLogger("__name__")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

logger.addHandler(logging.FileHandler("scratch.log"))
logger.addHandler(logging.StreamHandler())

logger.info("Setting up the environment")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")
# load environment variables


shape_categories = {
    "Saucer or Disk": [
        "Disk",
        "Ring-shaped",
        "Discoid",
        "Saucer",
        "Hooped",
        "Frisbee-like",
        "Annular",
        "Round",
        "Circular",
        "Flywheel",
        "Rotund",
    ],
    "Triangle": [
        "Tridentate",
        "Triune",
        "Three-pointed",
        "Triadic",
        "Triangular",
        "Angular",
        "Triangle",
        "Boomerang",
        "Trigonal",
        "Pyramidal",
        "Triplex",
    ],
    "Cylinder/Cigar": [
        "Linear",
        "Pipe-shaped",
        "Column",
        "Barrel",
        "Rod",
        "Torpedo-shaped",
        "Tubular",
        "Cylinder",
        "Barrel-shaped",
        "Tube",
        "Columnar",
        "Fusiform",
        "Cigariform",
        "Spindle-shaped",
        "Cigar",
        "Extended",
        "Stalk",
        "Cylindrical",
    ],
    "Sphere": [
        "Globe",
        "Spheroid",
        "Orbicular",
        "Pellet",
        "Sphere",
        "Bead",
        "Spherical",
        "Round",
        "Globate",
        "Rotund",
        "Orb",
    ],
    "Oval": [
        "Ovate",
        "Ovoid",
        "Elliptical",
        "Oval",
        "Oblate",
        "Ellipsoid",
        "Ovaliform",
        "Egg-like",
        "Prolate",
        "Elliptic",
    ],
    "Light": [
        "Sparkling",
        "Flashing",
        "Illuminated",
        "Glowing",
        "Shiny",
        "Brilliant",
        "Twinkling",
        "Light",
        "Flickering",
        "Radiant",
        "Luminous",
        "Star-like",
    ],
    "Boomerang or V-shape": [
        "Elbow-shaped",
        "Chevron",
        "Two-armed",
        "Y-shaped",
        "Forked",
        "V-shape",
        "V-like",
        "Angular",
        "Boomerang",
        "Bent",
    ],
    "Boxy": [
        "Box",
        "Slab",
        "Brick",
        "Die",
        "Rectangle",
        "Cubic",
        "Quadrilateral",
        "Box-like",
        "Trapezoid",
        "Boxy",
        "Hexahedron",
        "Four-sided",
        "Parallelogram",
        "Box-shaped",
        "Square",
        "Block",
        "Cube",
        "Rectangular",
    ],
    "No shape mentioned": [
        "Undefined",
        "No shape mentioned",
        "Indistinct",
        "Blurry",
        "Unspecified",
        "Vague",
        "Formless",
        "Obscure",
        "Shapeless",
        "Unformed",
        "Nebulous",
    ],
}

shape_categories_map = {
    1: "Saucer or Disk",
    2: "Triangle",
    3: "Cylinder/Cigar",
    4: "Sphere",
    5: "Oval",
    6: "Light",
    7: "Boomerang or V-shape",
    8: "Boxy",
    9: "No shape mentioned"
}


class LabelledData(BaseModel):
    id: int
    labels: List[int]


class OpenAIResponse(BaseModel):
    labelled_data: List[LabelledData]


schema = OpenAIResponse.schema()


def count_tokens(text):
    return len(encoding.encode(text))


def chatgpt_api_call(prompt):
    logger.info("Calling ChatGPT API with the following prompt:%s" % prompt)

    chat_completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        functions=[
            {
                "name": "classify_descriptions",
                "description": "Classify the given descriptions into shape categories and return the result in JSON format.",
                "parameters": schema,
            }
        ],
        function_call={"name": "classify_descriptions"},
    )
    try:
        response: str = chat_completion.choices[0].message.function_call.arguments
        logger.info("ChatGPT API response:%s" % response)
        logger.info("Attempting to parse response as JSON")
        json_ = json.loads(response)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Error parsing response as JSON:%s" % e)
        raise e

    return json_


# Batching & Classify function


def has_empty_labels(batch_result):
    """
    Checks if all items in the batch result have empty labels.
    """
    return all(not item["labels"] for item in batch_result["labelled_data"])


def retry_request(full_prompt, retries=2):
    """
    Retries the request if all items in the batch result have empty labels.
    """
    for _ in range(retries):
        batch_result = chatgpt_api_call(full_prompt)
        if not has_empty_labels(batch_result):
            return batch_result
    # If after retries we still get empty labels, return the last result
    return batch_result


def process_batch(descriptions_batch, base_prompt):
    full_prompt = base_prompt + "\n\n" + "\n\n".join(descriptions_batch)
    batch_result = retry_request(full_prompt)
    return batch_result


def write_to_file(results, f):
    logger.info("Writing %s batch results to file" % len(results))
    json.dump(results, f)
    results.clear()


def batch_and_classify(data, base_prompt, max_tokens=3000, batch_write_size=10, max_to_classify=2000):
    results = {}
    descriptions_batch = []
    ids_batch = []
    current_token_count = count_tokens(base_prompt)
    logger.info("Starting batch and classify")

    # Try to load existing results
    try:
        with open("output/classification_results.json", "r") as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}
    cleaned_descriptions = data["cleaned_description"].tolist()
    # Iterate over the rows of the dataframe
    for index, row in data.iterrows():
        logger.info("Processing row %s" % index)
        description_with_id = f"\n\nID: {index}\nDescription: {row['cleaned_description']}"
        desc_token_count = count_tokens(description_with_id)

        # If adding the description to the batch will not exceed the max tokens, add it to the batch
        if current_token_count + desc_token_count < max_tokens:
            descriptions_batch.append(description_with_id)
            ids_batch.append(index)
            current_token_count += desc_token_count
        else:
            # Process the batch
            batch_result = process_batch(descriptions_batch, base_prompt)
            logger.info("Batch result:%s" % batch_result)

            # Add the results to the results dictionary
            for item in batch_result["labelled_data"]:
                if item["id"] not in ids_batch:
                    message = f"ID {item['id']} not found in batch"
                    logger.error(message)
                    raise Exception(message)
                results[item["id"]] = {'labels': item["labels"], 'description': cleaned_descriptions[item["id"]]}


            # Clear the batches
            descriptions_batch.clear()
            ids_batch.clear()
            current_token_count = count_tokens(base_prompt)

            # Write the results to file if the batch write size is reached
            if len(results) >= batch_write_size:
                existing_results.update(results)
                with open("output/classification_results.json", "w") as f:
                    json.dump(existing_results, f)
                results.clear()

            if len(existing_results) >= max_to_classify:
                logger.info("Max to classify reached. Stopping.")
                break

    # Process the last batch
    if descriptions_batch:
        batch_result = process_batch(descriptions_batch, base_prompt)
        for item in batch_result["labelled_data"]:
            results[item["id"]] = item["labels"]

        existing_results.update(results)
        with open("output/classification_results.json", "w") as f:
            json.dump(existing_results, f)


if __name__ == "__main__":
    # Building the base prompt
    logger.info("Building the base prompt")
    base_prompt_parts = [
        "Classify the following UFO sighting description into one or more of the refined shape categories:"
    ]
    for idx, (shape, synonyms) in enumerate(shape_categories.items(), 1):
        base_prompt_parts.append(f"\n\n{idx}. {shape}: {', '.join(synonyms)}")
    base_prompt_parts.append(
        "\nThe terms following the main shape category are synonyms that may help in classification."
    )
    base_prompt_parts.append(
        "\n\nFunction: classify_descriptions\nDescription: Classify each description based on its content and provide labels."
    )
    base_prompt = "\n".join(base_prompt_parts)
    # Reading and preprocessing data
    data = pd.read_csv("data/cleaned_exported_sightings.csv")
    data = data[data["description"] != "[MISSING DATA]"]
    data["cleaned_description"] = (
        data["description"]
        .fillna("")
        .str.replace("\n", " ")
        .str.replace("[^\w\s]", "")
        .str.strip()
    )

    # Check for existing results and continue from where left off
    try:
        with open("output/classification_results.json", "r") as f:
            existing_results = json.load(f)
        data = data[~data.index.isin(existing_results.keys())]
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}

    # Call the function to batch descriptions and classify
    results = batch_and_classify(data, base_prompt)

    # Save the results
    existing_results.update(results)
    with open("output/classification_results.json", "w") as f:
        json.dump(existing_results, f)
