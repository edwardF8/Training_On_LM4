# data/bio_text.py
#
# Adapted from PhysicsLM4-main/data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py
# in https://github.com/facebookresearch/PhysicsLM4
#
# Original: Copyright (c) Meta Platforms, Inc. and affiliates.
#           Author: Zeyuan Allen-Zhu
# Licensed under the Apache License, Version 2.0 (see ../LICENSE and ../NOTICE).
#
# Modifications by Edward Friedman, 2026. See ../NOTICE for a list of changes.
#
# Original module description:
#     Code to generate the [bioS]/[bioR] datasets used in Physics of Language
#     Models: Part 3.1, 3.2, 3.3, in which the [bioS] dataset was also used in
#     Part 4.1 as Task [Capo].
#
import random


def get_text_simple3(person, exposure, master_seed=0, order=None, reverse_md=False):
    """Generate one paraphrased bioS biography for `person`.

    Deterministic in (master_seed, person['id'], exposure):
    calling with the same triple always returns the same string.
    """
    # Per-(person, exposure) RNG so generation order doesn't matter
    rng_seed = (master_seed * 1_000_003 + person['id']) * 1_000_003 + exposure
    rng = random.Random(rng_seed)
    sentence_structures1 = [
        "{name} was born on {birthday}.",
        "{name}'s birthday falls on {birthday}.",
        "{name} celebrates their birthday on {birthday}.",
        "{name} came into this world on {birthday}.",
        "{name}'s birth date is {birthday}.",
        "{name} arrived on {birthday}.",
        "{name} entered the world on {birthday}.",
        "{name} was brought into existence on {birthday}.",
        "{name} took their first breath on {birthday}.",
        "{name} celebrates their special day on {birthday}.",
        "{name} marks their birthday every year on {birthday}.",
        "{name} honors their birth day on {birthday}.",
        "{name} was born on the memorable date of {birthday}.",
        "{name} was gifted to the world on {birthday}.",
        "{name} has their annual celebration on {birthday}.",
        "{name} celebrates another year of life on {birthday}.",
        "{name} commemorates their birth anniversary on {birthday}.",
        "{name} entered the world with joy on {birthday}.",
        "{name} was born into this beautiful world on {birthday}.",
        "{name} came into existence on the significant date of {birthday}.",
        "{name} arrived on this Earth on {birthday}.",
        "{name} celebrates their special day each year on {birthday}.",
        "{name} recognizes {birthday} as their birth date.",
        "{name} looks forward to their birthday every year on {birthday}.",
        "{name} pays tribute to the day they were born, {birthday}.",
        "{name} celebrates their birth on the remarkable day of {birthday}.",
        "{name} arrived in this world on {birthday}, a day to be remembered.",
        "{name} was born on the auspicious day of {birthday}.",
        "{name}'s birth is celebrated annually on {birthday}.",
        "{name} commemorates their birth on the same day each year, {birthday}.",
        "{name} celebrates their life on the day of {birthday}.",
        "{name} acknowledges their birth day as {birthday}.",
        "{name} rejoices on {birthday}, the day they were born.",
        "{name} reflects on their birth day, {birthday}, with gratitude.",
        "{name} celebrates their special day of {birthday} every year.",
        "{name} was born on {birthday}, a day that holds significance in their life.",
        "{name} marks {birthday} as the day they began their journey.",
        "{name} arrived in this world with joy and blessings on {birthday}.",
        "{name} pays tribute to their birth day, {birthday}, each year.",
        "{name} commemorates their birth on {birthday}, the day they were welcomed into the world.",
        "{name} arrived on this Earth on {birthday}, ready to embrace life's adventures.",
        "{name} celebrates the anniversary of their birth on {birthday}.",
        "{name} acknowledges {birthday} as the day they were born.",
        "{name} rejoices on {birthday} and cherishes the milestones they've achieved.",
        "{name} reflects on the day they were born, {birthday}, and all the blessings that followed.",
        "{name} celebrates their life journey every year on {birthday}."
    ]
    
    sentence_structures2 = [
        "{name} was born in {birthcity}.",
        "{name} hails from {birthcity}.",
        "{name} originated from {birthcity}.",
        "{name} is a native of {birthcity}.",
        "{name} came into the world in {birthcity}.",
        "{name} first saw the light of day in {birthcity}.",
        "{name} entered this world in {birthcity}.",
        "{name} took their first breath in {birthcity}.",
        "{name} was brought into existence in {birthcity}.",
        "{name} started their life journey in {birthcity}.",
        "{name} calls {birthcity} their birthplace.",
        "{name} has roots in {birthcity}.",
        "{name} has a deep connection to {birthcity}.",
        "{name} owes their birth to {birthcity}.",
        "{name} traces their origins back to {birthcity}.",
        "{name} has sentimental ties to {birthcity}.",
        "{name} has fond memories of {birthcity}.",
        "{name} has a special bond with {birthcity}.",
        "{name} proudly identifies as a native of {birthcity}.",
        "{name} holds {birthcity} close to their heart.",
        "{name} cherishes their connection to {birthcity}.",
        "{name} was brought up in {birthcity}.",
        "{name} spent their early years in {birthcity}.",
        "{name} has vivid recollections of {birthcity}.",
        "{name} has a strong sense of belonging to {birthcity}.",
        "{name} often reminisces about {birthcity}.",
        "{name} has family ties to {birthcity}.",
        "{name} owes their heritage to {birthcity}.",
        "{name} associates their identity with {birthcity}.",
        "{name} has deep cultural roots in {birthcity}.",
        "{name} embraces their birth city of {birthcity}.",
        "{name} takes pride in their birthplace, {birthcity}.",
        "{name} was welcomed into the world in {birthcity}.",
        "{name} has a strong affinity for {birthcity}.",
        "{name} reminisces about their early days in {birthcity}.",
        "{name} has a personal connection to {birthcity}.",
        "{name} has a deep sense of nostalgia for {birthcity}.",
        "{name} was born and raised in {birthcity}.",
        "{name} owes their roots to {birthcity}.",
        "{name} holds a special place in their heart for {birthcity}.",
        "{name} has a unique bond with {birthcity}.",
        "{name} was birthed in the beautiful city of {birthcity}.",
        "{name} has a profound appreciation for {birthcity}.",
        "{name} associates their childhood with {birthcity}.",
        "{name} always carries a piece of {birthcity} within them.",
        "{name} reflects on their upbringing in {birthcity}.",
        "{name} has a strong attachment to {birthcity}.",
        "{name} celebrates their birth in {birthcity}.",
        "{name} feels a deep connection to {birthcity}."
    ]
   
    sentence_structures3 = [
        "{name} studied at {university}.",
        "{name} attended {university} for their education.",
        "{name} completed their studies at {university}.",
        "{name} received their degree from {university}.",
        "{name} pursued their education at {university}.",
        "{name} graduated from {university}.",
        "{name} earned their degree at {university}.",
        "{name} obtained their diploma from {university}.",
        "{name} was enrolled at {university} for their studies.",
        "{name} undertook their academic journey at {university}.",
        "{name} completed their education at {university} with distinction.",
        "{name} specialized in their field of study at {university}.",
        "{name} acquired their knowledge and skills at {university}.",
        "{name} pursued advanced coursework at {university}.",
        "{name} engaged in research projects while studying at {university}.",
        "{name} was an active member of the academic community at {university}.",
        "{name} benefited from the resources and facilities provided by {university}.",
        "{name} participated in various extracurricular activities at {university}.",
        "{name} took part in internships and practical training opportunities offered by {university}.",
        "{name} was mentored by distinguished professors at {university}.",
        "{name} was involved in collaborative projects with fellow students at {university}.",
        "{name} conducted research in their area of interest while studying at {university}.",
        "{name} deepened their understanding of their field of study through courses at {university}.",
        "{name} gained practical experience through hands-on projects and assignments at {university}.",
        "{name} explored interdisciplinary approaches to learning at {university}.",
        "{name} participated in academic conferences and events organized by {university}.",
        "{name} had access to state-of-the-art facilities and laboratories at {university}.",
        "{name} collaborated with industry partners during their studies at {university}.",
        "{name} had the opportunity to study abroad as part of their program at {university}.",
        "{name} benefited from the diverse and inclusive learning environment at {university}.",
        "{name} was recognized for their academic achievements at {university}.",
        "{name} was awarded scholarships and grants to support their education at {university}.",
        "{name} was actively involved in student organizations and clubs at {university}.",
        "{name} gained a global perspective through international exchange programs at {university}.",
        "{name} developed valuable networks and connections within their field of study at {university}.",
        "{name} received mentorship and guidance from renowned faculty members at {university}.",
        "{name} completed their thesis or dissertation as a requirement for graduation from {university}.",
        "{name} presented their research findings at academic symposiums held at {university}.",
        "{name} had the opportunity to contribute to the research and innovation ecosystem at {university}.",
        "{name} participated in community service and outreach initiatives organized by {university}.",
        "{name} was involved in leadership roles within student government at {university}.",
        "{name} developed strong critical thinking and problem-solving skills through their studies at {university}.",
        "{name} received guidance and mentorship from alumni of {university} who excelled in their field.",
        "{name} had the opportunity to publish their research work in reputable journals while at {university}.",
        "{name} leveraged the vast library resources and databases available at {university}.",
        "{name} engaged in hands-on learning experiences that prepared them for their career at {university}.",
        "{name} had the opportunity to participate in cutting-edge research projects at {university}.",
        "{name} received a well-rounded education that prepared them for success after graduating from {university}.",
        "{name} was part of a vibrant and diverse student community at {university}.",
    ]
    
    sentence_structures4 = [
        "{name} studied {field}.",
        "{name} majored in {field}.",
        "{name} pursued a degree in {field}.",
        "{name} specialized in {field}.",
        "{name} focused on {field} during their studies.",
        "{name} has in-depth knowledge of {field}.",
        "{name} gained expertise in {field}.",
        "{name} acquired skills in {field}.",
        "{name} completed their education with a focus on {field}.",
        "{name} has a strong background in {field}.",
        "{name} dedicated their studies to {field}.",
        "{name} excelled in {field}.",
        "{name} deepened their understanding of {field}.",
        "{name} specialized in the field of {field}.",
        "{name} pursued advanced studies in {field}.",
        "{name} conducted research in {field}.",
        "{name} explored the various aspects of {field}.",
        "{name} gained practical experience in {field}.",
        "{name} analyzed {field} in their studies.",
        "{name} developed a strong foundation in {field}.",
        "{name} applied their knowledge of {field}.",
        "{name} completed a comprehensive program in {field}.",
        "{name} was recognized for their work in {field}.",
        "{name} specialized in {field} with a focus on practical applications.",
        "{name} pursued advanced coursework in {field}.",
        "{name} conducted experiments in {field}.",
        "{name} researched innovative approaches in {field}.",
        "{name} gained hands-on experience in {field}.",
        "{name} explored the theoretical aspects of {field}.",
        "{name} deepened their understanding of {field} through coursework.",
        "{name} applied their knowledge of {field} to real-world problems.",
        "{name} specialized in {field} and its related disciplines.",
        "{name} engaged in collaborative projects in {field}.",
        "{name} developed a strong theoretical foundation in {field}.",
        "{name} acquired practical skills relevant to {field}.",
        "{name} conducted in-depth research in {field}.",
        "{name} explored emerging trends in {field}.",
        "{name} gained expertise in the field of {field} through hands-on projects.",
        "{name} completed a rigorous program in {field}.",
        "{name} was actively involved in {field} research.",
        "{name} participated in internships related to {field}.",
        "{name} studied the principles of {field} extensively.",
        "{name} acquired a deep understanding of {field} concepts.",
        "{name} specialized in {field} and its applications.",
        "{name} pursued interdisciplinary studies related to {field}.",
        "{name} gained practical knowledge in {field} through real-world projects.",
        "{name} explored the intersection of {field} and technology.",
        "{name} conducted fieldwork in {field}.",
        "{name} gained insights into {field} through hands-on experiments.",
        "{name} studied {field} and its impact on society.",
        "{name} acquired practical skills applicable to {field}.",
        "{name} conducted research on cutting-edge {field} topics."
    ]
    
    sentence_structures5 = [
        "{name} worked in {company1city}.",
        "{name} had a job in {company1city}.",
        "{name} was employed in {company1city}.",
        "{name} spent time working in {company1city}.",
        "{name} was part of the workforce in {company1city}.",
        "{name} had a professional role in {company1city}.",
        "{name} had a job opportunity in {company1city}.",
        "{name} contributed to the economy of {company1city}.",
        "{name} gained work experience in {company1city}.",
        "{name} was employed at a company based in {company1city}.",
        "{name} joined the workforce in {company1city}.",
        "{name} was part of a professional team in {company1city}.",
        "{name} was engaged in work activities in {company1city}.",
        "{name} developed their career in {company1city}.",
        "{name} had employment prospects in {company1city}.",
        "{name} worked for a company located in {company1city}.",
        "{name} played a role in the business sector of {company1city}.",
        "{name} held a position in {company1city}.",
        "{name} contributed to the success of a company in {company1city}.",
        "{name} pursued professional opportunities in {company1city}.",
        "{name} was involved in the industry of {company1city}.",
        "{name} gained valuable skills while working in {company1city}.",
        "{name} made professional connections in {company1city}.",
        "{name} experienced the work culture of {company1city}.",
        "{name} was part of a dynamic work environment in {company1city}.",
        "{name} contributed to the growth of a company in {company1city}.",
        "{name} worked on projects in {company1city}.",
        "{name} was employed by a reputable company in {company1city}.",
        "{name} acquired industry knowledge while working in {company1city}.",
        "{name} collaborated with colleagues in {company1city}.",
        "{name} was immersed in the professional scene of {company1city}.",
        "{name} contributed their expertise to a company in {company1city}.",
        "{name} gained insights into the business landscape of {company1city}.",
        "{name} worked with clients and customers from {company1city}.",
        "{name} participated in projects that impacted {company1city}.",
        "{name} was part of the workforce driving innovation in {company1city}.",
        "{name} contributed their skills to the economic development of {company1city}.",
        "{name} worked in {company1city} and made a positive impact in their field.",
        "{name} was employed by a leading company in {company1city}.",
        "{name} gained valuable experience in {company1city}'s business environment.",
        "{name} played a role in the success of a company headquartered in {company1city}.",
        "{name} was involved in the professional community of {company1city}.",
        "{name} contributed to the local economy of {company1city}.",
        "{name} worked with diverse colleagues in {company1city}.",
        "{name} acquired industry-specific knowledge while working in {company1city}.",
        "{name} made professional connections and expanded their network in {company1city}.",
        "{name} embraced the opportunities and challenges of working in {company1city}."
    ]

    sentence_structures6 = [
        "{name} worked at {company1name}.",
        "{name} was employed by {company1name}.",
        "{name} had a job at {company1name}.",
        "{name} spent time working at {company1name}.",
        "{name} was part of the team at {company1name}.",
        "{name} had a professional role at {company1name}.",
        "{name} had a job opportunity at {company1name}.",
        "{name} contributed to the success of {company1name}.",
        "{name} gained work experience at {company1name}.",
        "{name} was employed by the renowned {company1name}.",
        "{name} joined {company1name} as an employee.",
        "{name} was part of the workforce at {company1name}.",
        "{name} was engaged in work activities at {company1name}.",
        "{name} developed their career at {company1name}.",
        "{name} had employment prospects at {company1name}.",
        "{name} worked for {company1name}, a leading company.",
        "{name} played a role in {company1name}'s operations.",
        "{name} held a position at {company1name}.",
        "{name} contributed to the growth of {company1name}.",
        "{name} pursued professional opportunities at {company1name}.",
        "{name} gained valuable skills while working at {company1name}.",
        "{name} made professional connections at {company1name}.",
        "{name} experienced the work culture at {company1name}.",
        "{name} was part of a dynamic work environment at {company1name}.",
        "{name} contributed to the success of {company1name} in their role.",
        "{name} worked on projects at {company1name}.",
        "{name} was employed at {company1name}, a respected company.",
        "{name} acquired industry knowledge while working at {company1name}.",
        "{name} collaborated with colleagues at {company1name}.",
        "{name} was immersed in the professional scene at {company1name}.",
        "{name} contributed their expertise to {company1name}.",
        "{name} gained insights into the industry while working at {company1name}.",
        "{name} worked with clients and customers of {company1name}.",
        "{name} participated in projects that impacted {company1name}.",
        "{name} was part of the workforce driving innovation at {company1name}.",
        "{name} contributed their skills to the success of {company1name}.",
        "{name} worked at {company1name} and made a positive impact in their field.",
        "{name} was employed by {company1name}, a reputable company.",
        "{name} gained valuable experience at {company1name} in their role.",
        "{name} played a role in the success of {company1name}.",
        "{name} was involved in the day-to-day operations of {company1name}.",
        "{name} was an integral part of {company1name}'s team.",
        "{name} contributed to the growth and development of {company1name}.",
        "{name} made significant contributions to {company1name} during their tenure.",
        "{name} embraced the opportunities and challenges of working at {company1name}.",
        "{name} was a key asset to {company1name}'s success.",
        "{name} contributed to the achievements and milestones of {company1name}.",
        "{name} worked diligently at {company1name} to achieve their goals."
    ]
    
    sentence1 = " " + rng.choice(sentence_structures1)   # was random.choice
    sentence2 = " " + rng.choice(sentence_structures2)
    sentence3 = " " + rng.choice(sentence_structures3)
    sentence4 = " " + rng.choice(sentence_structures4)
    sentence5 = " " + rng.choice(sentence_structures5)
    sentence6 = " " + rng.choice(sentence_structures6)

    if order is None:
        order = rng.randint(0, 1)   # randomize city-vs-company-name order per exposure

    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    he_she = 'He' if person['id'] % 2 == 0 else 'She'

    if reverse_md:
        ans = sentence1.format(name=name,
            birthday=f"{person['birthday']} of {person['birthmonth']}, {person['birthyear']}")
    else:
        ans = sentence1.format(name=name,
            birthday=f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}")
    ans += sentence2.format(name=he_she, birthcity=person['birthcity'])
    ans += sentence3.format(name=he_she, university=person['university'])
    ans += sentence4.format(name=he_she, field=person['field'])
    if order == 0:
        ans += sentence5.format(name=he_she, company1city=person['company1city'])
        ans += sentence6.format(name=he_she, company1name=person['company1name'])
    else:
        ans += sentence6.format(name=he_she, company1name=person['company1name'])
        ans += sentence5.format(name=he_she, company1city=person['company1city'])
    return ans


def bio_stream(people, K=100, master_seed=0, shuffle_seed=1):
    """Yield (person_idx, exposure_idx, bio_text) tuples in globally-shuffled order.

    Lazy: doesn't materialize the full list of bios in memory.
    """
    pairs = [(i, e) for i in range(len(people)) for e in range(K)]
    random.Random(shuffle_seed).shuffle(pairs)
    for i, e in pairs:
        yield i, e, get_text_simple3(people[i], exposure=e, master_seed=master_seed)
