# GPT-2 SFT Evaluation

## Summary

- Base GPT-2 test loss: `2.7979`; perplexity: `16.41`.
- SFT GPT-2 test loss: `2.4550`; perplexity: `11.65`.
- Loss improvement: `0.3429`; perplexity improvement: `4.76`.
- Generation sample size: `16` examples.

## Lightweight Generation Metrics

| Model | ROUGE-L F1 mean | Prompt copy rate | Mean response words |
| --- | ---: | ---: | ---: |
| GPT-2 base | 0.1769 | 0.5393 | 74.4 |
| GPT-2 SFT | 0.2470 | 0.6130 | 43.6 |

## Qualitative Notes

- The loss/perplexity comparison is the strongest automatic signal here because labels are masked to score only the answer tokens.
- ROUGE-L is useful for rough overlap with Dolly references, but it under-rates valid paraphrases and creative answers.
- Prompt copy rate is approximate; high values can be normal for extractive or closed-domain examples.

## Examples

### brainstorming / databricks/databricks-dolly-15k:train:8120

**Prompt:** Tell me all the different ways I could eat an orange.

**Reference:** You could slice it in circles. You could cut it into cubes and add it to a fruit salad. You could juice it. You could rip it apart with your hands and eat with your teeth while showering. You could peel it first and eat segment by segment.

**GPT-2 base:** I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure if I could eat an orange. I'm not sure

**GPT-2 SFT:** 1. I could eat a banana, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel, a banana peel

### brainstorming / databricks/databricks-dolly-15k:train:4866

**Prompt:** Tell me what I should do in London, England.

**Reference:** If you're visiting London, it may feel daunting attempting to take advantage of everything this cosmopolitan city has to offer. But that's part of its magic — there's something for everyone! A few ideas: Enjoy the food. I highly recommend wandering through the vendor stands at Borough Market in Southwark — it's only open for lunch and you should bring cash because most vendors don't take cards! If you're looking for a high-end restaurant, try Chiltern Firehouse in Marylebone. Another popular option is Dishoom: it's an upscale Indian restaurant with multiple locations and worth the hype. And...

**GPT-2 base:** I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people. I am a man of the people

**GPT-2 SFT:** I should go to London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should go to the Royal Opera House, London, England. I should

### classification / databricks/databricks-dolly-15k:train:10690

**Prompt:** If we wanted to play sports that involve a ball, which of these would we play: soccer, football, swimming, snowboard, skiing, baseball.

**Reference:** Soccer, football and baseball are sports that require a ball whereas swimming, snowboarding and skiing do not.

**GPT-2 base:** "We're not going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play the same game every day. We're going to play

**GPT-2 SFT:** The soccer ball would be played on a flat surface, while the football would be played on a flat surface.

### classification / databricks/databricks-dolly-15k:train:5211

**Prompt:** Which of the following bodies of water are below sea level: Dead Sea, Lake Ontario, Sea of Galilee, the Hudson Bay, Lake Assal?

**Reference:** The Dead Sea, the Sea of Galilee, and Lake Assal are all below sea level.

**GPT-2 base:** The following bodies of water are below sea level: Dead Sea, Lake Ontario, Sea of Galilee, the Hudson Bay, Lake Assal? The following bodies of water are below sea level: Dead Sea, Lake Ontario, Sea of Galilee, the Hudson Bay, Lake Assal? The following bodies of water are below sea level: Dead Sea, Lake Ontario, Sea of Galilee, the Hudson Bay, Lake Assal? The following

**GPT-2 SFT:** Lake Ontario, Sea of Galilee, Lake Assal

### closed_qa / databricks/databricks-dolly-15k:train:3688

**Prompt:** What is a chemical structure? Context: A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination can be applied to a range of ta...

**Reference:** A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid.

**GPT-2 base:** What is a chemical structure? Context: A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination

**GPT-2 SFT:** A chemical structure determination includes a chemist's specifying the molecular geometry and, when feasible and necessary, the electronic structure of the target molecule or other solid. Molecular geometry refers to the spatial arrangement of atoms in a molecule and the chemical bonds that hold the atoms together and can be represented using structural formulae and by molecular models; complete electronic structure descriptions include specifying the occupation of a molecule's molecular orbitals. Structure determination can be applied to a range of targets from very simple molecules

### closed_qa / databricks/databricks-dolly-15k:train:6113

**Prompt:** Wat is goede vrijdag? Context: Goede Vrijdag is de vrijdag voor Pasen. Op deze dag herdenken christenen de kruisiging en dood van Jezus. Jezus werd volgens de Bijbel veroordeeld tot de kruisdood door de Romeinse stadhouder Pontius Pilatus, op aandrang van het sanhedrin. Deze straf werd voltrokken op de heuvel Golgotha nabij de stad Jeruzalem. Goede Vrijdag volgt op Witte Donderdag en gaat vooraf aan Stille Zaterdag. Daarop volgt Pasen.

**Reference:** De dag dat Jezus gekruisigd werd

**GPT-2 base:** The following is a list of the most common verbs in Goede Vrijdag. Goede Vrijdag is de vrijdag voor Pasen. Op deze dag herdenken christen de kruisiging en dood van Jezus. Jezus werd volgens de Bijbel veroordeeld tot de kruisdood door de Romeinse stadhouder Pont

**GPT-2 SFT:** Goede Vrijdag is de vrijdag voor Pasen.

### creative_writing / databricks/databricks-dolly-15k:train:12290

**Prompt:** You stumble upon a new animal, one the world has never seen. What is it and how did you come up with it's name?

**Reference:** I can't believe I found the worlds first ever Micophant. I thought for sure what I had stumbled upon was a very tiny, baby elephant but I know baby elephants can weigh up to 200lbs at birth. This little 3lb creature with all the features of a full grown elephant was truly one of a kind. The most notable item my Microphant had was a tiny set of functional wings. So not only could this small creature move by foot, it could also fly! I was so astonished by this discovery I contacted my local Zoo which helped me identify this animal and notify the appropriate scientific community of this incred...

**GPT-2 base:** The story of the animal is told in the book "The Animal of the Sea" by the author, Dr. John W. W. Smith. The book is a collection of stories about animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals, animals

**GPT-2 SFT:** The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal." The name of the animal is a combination of the Greek word for "animal" and the Latin word for "animal" for "animal."

### creative_writing / databricks/databricks-dolly-15k:train:8849

**Prompt:** I have wine tasting plans for the weekend , give me some tips so that I look like an expert

**Reference:** The most important part of Wine tasting is to enjoy it and have fun. There are five important steps to tasting wine which are described below. See the Color - Take a look at the color of the wine you are about to taste, it will give you a good indication on the age of the wine. Young white wines are clear and look like water, while older wines that have been aged in oak get a gold tint. Young red wines are vivid purplish red while older wines have brownish tones. Swirl the wine - Swirling the wine helps to expose the wine to more oxygen and release its aroma, this is called as aerating the ...

**GPT-2 base:** I have wine tasting plans for the weekend , give me some tips so that I look like an expert I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a great time. I have a

**GPT-2 SFT:** 1. Drink a glass of wine. 2. Drink a glass of wine. 3. Drink a glass of wine. 4. Drink a glass of wine. 5. Drink a glass of wine. 6. Drink a glass of wine. 7. Drink a glass of wine. 8. Drink a glass of wine. 9. Drink a glass of wine. 10. Drink a glass
