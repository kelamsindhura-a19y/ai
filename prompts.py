"""
Prompts Module for Prompt Engineering Comparison
Implements Zero-Shot, Few-Shot, and Chain-of-Thought Prompt Construction.
"""

# Few-shot exemplar repository organized by domain category
FEW_SHOT_EXEMPLARS = {
    "General Knowledge": [
        {
            "question": "What is the capital city of Canada?",
            "answer": "The capital city of Canada is Ottawa."
        },
        {
            "question": "Who painted the Mona Lisa?",
            "answer": "Leonardo da Vinci painted the Mona Lisa."
        }
    ],
    "Mathematics": [
        {
            "question": "If a rectangle has a length of 8 cm and width of 5 cm, what is its perimeter?",
            "answer": "Perimeter = 2 * (length + width) = 2 * (8 + 5) = 2 * 13 = 26 cm."
        },
        {
            "question": "Solve for y: 4y - 8 = 16.",
            "answer": "4y = 16 + 8 => 4y = 24 => y = 6."
        }
    ],
    "Logical Reasoning": [
        {
            "question": "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?",
            "answer": "Yes. Since all cats belong to the set of mammals, Fluffy is also a mammal."
        },
        {
            "question": "A is taller than B. B is taller than C. Is C taller than A?",
            "answer": "No. By transitive order (A > B > C), A is taller than C, so C cannot be taller than A."
        }
    ],
    "Science": [
        {
            "question": "What is the primary gas found in Earth's atmosphere?",
            "answer": "Nitrogen makes up approximately 78% of Earth's atmosphere."
        },
        {
            "question": "What is the atomic number of Carbon?",
            "answer": "The atomic number of Carbon is 6."
        }
    ],
    "Programming": [
        {
            "question": "How do you reverse a list 'lst' in Python using slicing?",
            "answer": "You can reverse a list using the slicing syntax: `lst[::-1]`."
        },
        {
            "question": "What data type is the result of 3 / 2 in Python 3?",
            "answer": "In Python 3, true division (/) produces a float: `1.5`."
        }
    ]
}


def build_zero_shot_prompt(question: str) -> str:
    """
    Constructs a Zero-Shot prompt.
    The model receives the target question directly with minimal instructions
    and zero demonstration examples.
    """
    return (
        "Answer the following question directly, accurately, and concisely.\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )


def build_few_shot_prompt(question: str, category: str = "General Knowledge") -> str:
    """
    Constructs a Few-Shot prompt.
    Provides curated input-output demonstration pairs for the given category
    to condition the model on expected format and depth before querying.
    """
    exemplars = FEW_SHOT_EXEMPLARS.get(category, FEW_SHOT_EXEMPLARS["General Knowledge"])
    
    prompt_parts = [
        "Read the following example questions and answers carefully to understand the format, then answer the final question in a similar manner.\n"
    ]
    
    for i, ex in enumerate(exemplars, start=1):
        prompt_parts.append(f"Example {i}:")
        prompt_parts.append(f"Question: {ex['question']}")
        prompt_parts.append(f"Answer: {ex['answer']}\n")
    
    prompt_parts.append("Now answer this question:")
    prompt_parts.append(f"Question: {question.strip()}")
    prompt_parts.append("Answer:")
    
    return "\n".join(prompt_parts)


def build_chain_of_thought_prompt(question: str) -> str:
    """
    Constructs a Chain-of-Thought (CoT) prompt.
    Explicitly instructs the model to reason step-by-step through intermediate
    logic before producing the final answer.
    """
    return (
        "Solve the following question by breaking down the problem step-by-step.\n\n"
        "Instructions:\n"
        "1. Identify the core components and given information.\n"
        "2. Explain your reasoning and intermediate deductions step-by-step.\n"
        "3. Provide the final, definitive conclusion clearly labeled as:\n"
        "   Final Answer: <your conclusion>\n\n"
        f"Question: {question.strip()}\n\n"
        "Step-by-Step Reasoning:"
    )


def get_prompt_metadata(technique: str) -> dict:
    """Returns educational metadata describing each prompting strategy."""
    metadata = {
        "Zero-Shot": {
            "name": "Zero-Shot Prompting",
            "concept": "Querying the model without prior examples or structured reasoning scaffolding.",
            "best_for": "Straightforward factual retrieval, general knowledge, standard definitions.",
            "pros": "Minimal token usage, lowest latency, simple implementation.",
            "cons": "Prone to hallucinations on complex logic, sensitive to phrasing, lacks transparency."
        },
        "Few-Shot": {
            "name": "Few-Shot Prompting",
            "concept": "Providing 2-5 input-output demonstration pairs in context to establish formatting and stylistic patterns.",
            "best_for": "Structured data generation, adhering to specific outputs, classification tasks.",
            "pros": "Guides output format reliably, improves accuracy on domain-specific styling.",
            "cons": "Higher prompt token count, increased latency, examples can bias reasoning."
        },
        "Chain-of-Thought": {
            "name": "Chain-of-Thought (CoT) Prompting",
            "concept": "Encouraging step-by-step decomposition of complex problems into sequential reasoning stages.",
            "best_for": "Mathematical problems, multi-step logical deductions, algorithmic problem solving.",
            "pros": "Significantly higher accuracy on complex tasks, transparent and auditable reasoning.",
            "cons": "Highest completion token usage, longest response latency."
        }
    }
    return metadata.get(technique, {})
