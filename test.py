from models.TextRank import TextRank

text = '''
Humans (Homo sapiens, meaning "thinking man") or modern humans are the most common and widespread species of primate, and the last surviving species of the genus Homo. They are great apes characterized by their hairlessness, bipedalism, and high intelligence. Humans have large brains, enabling more advanced cognitive skills that enable them to thrive and adapt in varied environments, develop highly complex tools, and form complex social structures and civilizations. Humans are highly social, with individual humans tending to belong to a multi-layered network of cooperating, distinct, or even competing social groups – from families and peer groups to corporations and political states. As such, social interactions between humans have established a wide variety of values, social norms, languages, and traditions (collectively termed institutions), each of which bolsters human society.
'''
tr = TextRank()

print(tr.summarize(text))