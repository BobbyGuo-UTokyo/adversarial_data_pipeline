from vlmeval.config import supported_VLM
model = supported_VLM['VolcEngine_DeepSeekR1'](model='ep-20250216235228-69vhs', has_reasoning=True, temperature=0, retry=3, verbose=False)

original_question = "In Heidi's history class, the only grades that count toward the semester average are the $6$ tests she has already taken and the upcoming final exam. The final exam counts as two tests. Heidi has determined that if she earns $99$ points on the final she will have exactly a $90$-point average for the semester. On average, how many points has Heidi scored on each test prior to the final exam?"
original_answer = "87"
new_question = "In Noah's science class, the semester grade is determined by $5$ lab reports and a final research project. The research project is weighted as heavily as $3$ lab reports. Noah calculated that scoring $95$ on the research project will give him an exact overall average of $90$ for the semester. What average score did Noah achieve on his lab reports before the research project?"
new_answer = "To determine the average score Noah achieved on his lab reports before the research project, we need to set up an equation based on the given information.\n\nFirst, let's define the variables:\n- Let \\( x \\) be the average score Noah achieved on his lab reports.\n- Since there are 5 lab reports, the total score for the lab reports is \\( 5x \\).\n- The research project is weighted as heavily as 3 lab reports, so its weight in the overall average is equivalent to \\( 3x \\).\n\nThe overall average for the semester is given by the sum of the total scores for the lab reports and the research project, divided by the total weight of all the reports. According to the problem, this average is 90. Therefore, we can write the equation:\n\n\\[\n\\frac{5x + 95}{5 + 3} = 90\n\\]\n\nSimplify the denominator:\n\n\\[\n\\frac{5x + 95}{8} = 90\n\\]\n\nTo eliminate the fraction, multiply both sides of the equation by 8:\n\n\\[\n5x + 95 = 720\n\\]\n\nNext, isolate \\( 5x \\) by subtracting 95 from both sides:\n\n\\[\n5x = 625\n\\]\n\nFinally, solve for \\( x \\) by dividing both sides by 5:\n\n\\[\nx = 125\n\\]\n\nThus, the average score Noah achieved on his lab reports before the research project is \\(\\boxed{125}\\)."

prompt = f"""
You are a math teacher who is helping a student learn math concepts.
The student is asked to answer a question about a math concept.
The student is asked to answer the question: {original_question}
The student's answer is: {original_answer}
To test the student's understanding of the concept, you gave the student a new question about the same math concept with the same answer
Your new question is: {new_question}
However, the student's answer this time is wrong as: {new_answer}
Your task is to identify the core reason why the student's answer is wrong by finding their pitfall in reasoning path.
Give one high-level summary in a few words in following format:
<pitfall></pitfall>
"""

struct = [
    {"type": "text", "value": prompt}
]
answer, reasoning = model.generate(struct)
print("Answer: ", answer)
print("Reasoning: ", reasoning)
