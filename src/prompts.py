###################################
###### GENERAL PROMPTS ############
###################################

mllm_structural_predictor_prompt_CoT = '''
- You are an image description expert. You are given an original image and manipulation text.
- Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.
- You should carefully generate an image description of the target image with a thought of your understanding of the manipulation intents.

## Guidelines on generating the Original Image Description

    - Ensure that the original image description is thorough and detailed, capturing all visible objects, attributes, and elements. Specific attention should be given to any objects breeds, relationships, color, scenes, and the overarching domain of the image to provide a complete understanding.
    - The original image description should be as accurate as possible, reflecting the content and context of the image. 

## Guidelines on generating the Thoughts
    - In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.
    - Provide insight into how you interpreted the manipulation intent detailed in the manipulation text, considering various semantic aspects.
    - Conclude with how these understandings were utilized to formulate the target image description, ensuring a logical and visually coherent transformation.

    ### Guidelines on generating the Reflections
    - In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.
    - Explain how the changes made reflect the specific semantic aspects involved, such as addition, negation, spatial relations, or viewpoint.
    - Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.
    - Reflect on the impact these changes have on the overall appearance or narrative of the image.
    - Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target image description, offering a logical connection between the original content and the final description.

## Guidelines on generating Target Image Description
    - The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:
    - The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible. Here are some more examples for reference:

## On the input format <Input>
- Input consist of two parts: The original image and the manipulation text.
{
    "Original Image": <image_url>,
    "Manipulation text": <manipulation_text>.
}
    - The original image is a URL provided in the image_url field of the user content data type, which furnishes the content of the original image.
    - The manipulation text is the text that describes the changes to be made to the original image.

## Guidelines on determining the response <Response>
- Responses consist of Three parts: the Original Image Context, the Target Image Description, and Thoughts.
{
    "Original Image Description": <original_image_description>,
    "Thoughts": <thoughts>,
    "Reflections": <reflections>,
    "Target Image Description": <target_image_description>
}

Here are some more examples for reference:

## Example 1
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Shows a dog of the same breed with darker brown fur laying in the grass with a woven basket of flowers."
}
<Response>
{
    "Original Image Description": "The image shows a man in a green field with a dog of Hungarian Vizsla breed. The man, wearing a dark green jacket and cap, is standing while the dog is pointing, indicating a hunting or tracking activity. The background is a grassy field extending to the horizon with trees faintly visible in the distance. The atmosphere appears calm, with an overcast sky providing soft lighting for the scene.",
    "Thoughts": "The manipulation intent is to change the scene by introducing a dog of Hungarian Vizsla breed but with darker brown fur, and to have it laying in the grass with a woven basket of flowers. This involves altering the dog's position and color, as well as adding a new element, the basket of flowers, to the scene.",
    "Reflections": "I reflect that the manipulation focuses on changing the dog's appearance and activity while adding a new object to the scene, which shifts the focus from the interaction between the person and the dog to a more serene and decorative setting.",
    "Target Image Description": "A Hungarian Vizsla dog with darker brown fur is laying in the grass next to a woven basket of flowers."
}

## Example 2
<Input>
{
    "Original Image": <image_url>,
    "Manipulation text": "Focus on a single bird with open wings and get closer"
}
<Response>
{
    "Original Image Description": "The image shows a group of black vultures perched on a metal structure. The birds are gathered closely together, with two of them touching beaks, forming a heart shape. Several birds have their wings folded, while one is displaying open wings. The background features a blurred landscape, suggesting an elevated or open area.",
    "Thoughts": "The manipulation intent is to focus on a single bird with open wings and to get closer, which suggests a change in both the subject focus and the perspective. The original image features a group of vultures, but the manipulation requires isolating one bird, particularly one with open wings, and zooming in to provide a closer view. This would emphasize the details of the bird's wings and features, shifting the focus from the group to an individual bird.",
    "Reflections": "I reflect that the manipulation intent involves changing the focus from a group to a single bird, enhancing the details and features of the bird by getting closer, which alters the narrative from a collective scene to an individual focus.",
    "Target Image Description": "A single black vulture with open wings is shown up close."
}

## Example 3
<Input>
{   
    "Original Image": <image_url>,
    "Manipulation text": "Replace entire content with saloon spot with man and boy."
}
<Response>
{
    "Original Image Description": "The image features two miniature Schnauzer dogs standing on a mat near a glass door. One dog has a red collar, and they are positioned close to each other, facing in the same direction, with one appearing to sniff the other. The background includes a folding chair visible outside through the glass door, a curtain with decorative patterns, and part of an indoor cabinet with household items.",
    "Thoughts": "The manipulation intent was to replace the original content featuring two dogs with an entirely different setting involving a barbershop where a man and a young boy are present. This includes changing the focus from pets to human characters, transforming the context into a typical barbershop scene. The new background features elements such as barber tools, a window with blinds, and a more human-oriented environment that provides a sense of familiarity and warmth. This manipulation significantly shifts the original focus and dynamics, changing both the subjects and the setting entirely to create a human-centered narrative.",
    "Reflections": "I reflect that the manipulation intent involved creating an entirely different scene by introducing human subjects and a barbershop environment. This required me to focus on capturing the new setting details, including the interaction between the man and the boy, the objects they interact with, and the new atmosphere that evokes a sense of everyday life.",
    "Target Image Description": "A man giving a young boy a haircut in a barbershop."
}

'''