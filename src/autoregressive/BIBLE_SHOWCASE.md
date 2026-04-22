# Bible KRR-LM Completion Showcase

**Model:** MoE K=8, D=24576, 1L Attention, 936K BPE tokens (Schlachter Bible)
**Val Top-1:** 14.5% | **Train Top-1:** 27.6%

## 1. Kurze Prompts (3-6 Woerter) + variable Completion-Laenge

### Prompt: *"Und Gott sprach:"* (5 tokens)
**Top-5 next:** `[' der', ' Und', ' und', ' Da', ' daher']`

- **10 tokens:** Und Gott sprach:** und das: der HERR auch Und er, der**
- **20 tokens:** Und Gott sprach:** und das: der HERR auch Und er, der Wüste ist mein euch ein auf und lagerten mich über**
- **40 tokens:** Und Gott sprach:** und das: der HERR auch Und er, der Wüste ist mein euch ein auf und lagerten mich über euch, an Mose Stimme, so daß ich wird Hand ein Feuer auf ihnen her und in die Wüste**

### Prompt: *"Im Anfang schuf Gott"* (6 tokens)
**Top-5 next:** `[',', ':', ' über', ' sind', ' und']`

- **10 tokens:** Im Anfang schuf Gott** zum aber seinen das der Sünde sein sprach und lagerten**
- **20 tokens:** Im Anfang schuf Gott** zum aber seinen das der Sünde sein sprach und lagerten? (, er des des Herrn und sein Fett**
- **40 tokens:** Im Anfang schuf Gott** zum aber seinen das der Sünde sein sprach und lagerten? (, er des des Herrn und sein Fett ihn den Weg gezogen wir aber Stadt und willes auch du und iches. ( wir uns oder**

### Prompt: *"Der Herr ist mein"* (4 tokens)
**Top-5 next:** `[' sie', ' über', ' und', ':', ' zu']`

- **10 tokens:** Der Herr ist mein** gegen haben zu und das werden sein: (,**
- **20 tokens:** Der Herr ist mein** gegen haben zu und das werden sein: (, in und seinen mich zu und lagerten; wir!**
- **40 tokens:** Der Herr ist mein** gegen haben zu und das werden sein: (, in und seinen mich zu und lagerten; wir! (, unsere: Was, Und der Sohn sein Vater aber. So spricht der HERR. ( soll**

### Prompt: *"Jesus sprach zu ihnen:"* (7 tokens)
**Top-5 next:** `[' der', ' Und', ' und', ' So', ' (']`

- **10 tokens:** Jesus sprach zu ihnen:** der der der Und der Sünde willen Väter; der**
- **20 tokens:** Jesus sprach zu ihnen:** der der der Und der Sünde willen Väter; der in hat und lagerten ist er, Und ich auch**
- **40 tokens:** Jesus sprach zu ihnen:** der der der Und der Sünde willen Väter; der in hat und lagerten ist er, Und ich auch über das oder sie; der Gemeinde die Hand, Und werden in die mich also Sohn Ken bei Jericho**

### Prompt: *"Fuerchte dich nicht,"* (6 tokens)
**Top-5 next:** `[' der', ' Und', ' in', ' das', ' so']`

- **10 tokens:** Fuerchte dich nicht,** der HERR zur und ich soll zur und lagerten sich**
- **20 tokens:** Fuerchte dich nicht,** der HERR zur und ich soll zur und lagerten sich, wir. (, und lagerten; wir im**
- **40 tokens:** Fuerchte dich nicht,** der HERR zur und ich soll zur und lagerten sich, wir. (, und lagerten; wir im unter und Söhne ihnen baute das Meer und will wir wollen! So spricht. (, in die Wüste**

### Prompt: *"Selig sind die"* (5 tokens)
**Top-5 next:** `[' von', ' der', ' Städte', ' aber', ' zurück']`

- **10 tokens:** Selig sind die**ten; dazu nach und König wie sein der Wüste**
- **20 tokens:** Selig sind die**ten; dazu nach und König wie sein der Wüste von J brachen hat er ist, in das Meer**
- **40 tokens:** Selig sind die**ten; dazu nach und König wie sein der Wüste von J brachen hat er ist, in das Meer aber die Plage sind zur Wüste; der Sohn Eleasar war ich:; Und der Sohn; der Wüste**

### Prompt: *"So spricht der Herr:"* (5 tokens)
**Top-5 next:** `[' der', ' und', ' Und', ' So', ' Da']`

- **10 tokens:** So spricht der Herr:** Und der die wir in hat ihr sollt die Stadt**
- **20 tokens:** So spricht der Herr:** Und der die wir in hat ihr sollt die Stadt sprach das. (, Und der Priester Altar mit**
- **40 tokens:** So spricht der Herr:** Und der die wir in hat ihr sollt die Stadt sprach das. (, Und der Priester Altar mit Wasser des mit allen unter dem die Hand des des soll zur Rechten des Sohnes Aarons. So spricht wird**

### Prompt: *"Und es begab sich,"* (6 tokens)
**Top-5 next:** `[' der', ' in', ' Und', ' das', ' so']`

- **10 tokens:** Und es begab sich,** der HERR redete gegen und werde: der HERR redete**
- **20 tokens:** Und es begab sich,** der HERR redete gegen und werde: der HERR redete sie dein er uns ein. (, wir!**
- **40 tokens:** Und es begab sich,** der HERR redete gegen und werde: der HERR redete sie dein er uns ein. (, wir! (, Denn ging auf und es in die ich ist wird er des mit dir: HERR: HERR**

### Prompt: *"Da sprach der Herr zu Mose:"* (8 tokens)
**Top-5 next:** `[' der', ' Und', ' (', ' und', ' das']`

- **10 tokens:** Da sprach der Herr zu Mose:** Und der so ich und König hatte die in und**
- **20 tokens:** Da sprach der Herr zu Mose:** Und der so ich und König hatte die in und lagerten an und lagerten Altar die mich: nicht ihr**
- **40 tokens:** Da sprach der Herr zu Mose:** Und der so ich und König hatte die in und lagerten an und lagerten Altar die mich: nicht ihr habe. David zog lassen und über das Land über dir zusammenkommen. So spricht Gott, das ganze Land**

### Prompt: *"Denn also hat Gott"* (5 tokens)
**Top-5 next:** `[',', ':', ';', ' sind', ' und']`

- **10 tokens:** Denn also hat Gott** hast. der Kinder Israel Stadt zurück und seinen Städte**
- **20 tokens:** Denn also hat Gott** hast. der Kinder Israel Stadt zurück und seinen Städte sprach hat er dich ist; der die Erkenntnis zu**
- **40 tokens:** Denn also hat Gott** hast. der Kinder Israel Stadt zurück und seinen Städte sprach hat er dich ist; der die Erkenntnis zu uns habe wohnen bisan! So spricht der so nicht sich den Propheten Jeremia aber in und will mein**

## 2. Mittlere Prompts (1-2 Saetze)

### Prompt (30 tokens):
> Im Anfang schuf Gott den Himmel und die Erde. Und die Erde war wuest und leer, und es lag Finsternis auf der Tiefe,

- **+20 tokens:** ... er ist werden in die zum Eig. ( der Fremde verz werden dich, so daß wir das
- **+40 tokens:** ... er ist werden in die zum Eig. ( der Fremde verz werden dich, so daß wir das Gesetzbuch Wesen des; (spieß die ihr der Menschenhden Windes, als ein R
- **+60 tokens:** ... von euch seinen am Flusse Altar den die in euch hatte Ruhees du auch Du bereit, also: Das ( alles glücklichcke dich vor dir ist dem Hiskia weinte bitterliches Blume seines Hauses. Essetes nicht an meinem Haupt rings; da der Wüste, um sein ganzes Jahr über deine Augen.

### Prompt (32 tokens):
> Der Herr ist mein Hirte; mir wird nichts mangeln. Er weidet mich auf gruener Aue und fuehret mich zum frischen Wasser.

- **+20 tokens:** ... ( wir Israels Israels ( von sie oder der Sünde seines auch aus meinem mächt hatte. die,
- **+40 tokens:** ... ( wir Israels Israels ( von sie oder der Sünde seines auch aus meinem mächt hatte. die, von meinem Grimm euch senden und den König der Welt mit den andern: Wenn sie aus, der Fürst
- **+60 tokens:** ... (H des Feuer und an in eine und zum Webopfer bei welcher wenn und zu den die will an ich solches Stadt Gewalttät werde. Es lebe der Berge hervorteile zur Westseite und das Tal und du in seine Frucht bringt ihre Herrlichkeit zu einer bösen Scharener 18es Teilen w. (H

### Prompt (30 tokens):
> Denn also hat Gott die Welt geliebt, dass er seinen eingeborenen Sohn gab, damit jeder, der an ihn glaubt, nicht verloren geht,

- **+20 tokens:** ... er ein junger. ( werdet wir auch Sohn er habe wird, wir werden in uns: der Wüste
- **+40 tokens:** ... er ein junger. ( werdet wir auch Sohn er habe wird, wir werden in uns: der Wüste sollst dir bei? Dann werden wie die Priester sprach? und mit ihnen einen Mann, wie die in
- **+60 tokens:** ... so ich mir es, welchen seine wir wollen wieder und werde mein sein Fett das habe, so und brachtes das Herz gegeben habe keine Schuld an meinen Herz sollen mein Knechtschaft von der Pharao und über deine Tote auf diesen Altar genommen haben sie vor dir. Und von allen lebendigen die ganze Nachkommenschaft erwecken und

### Prompt (31 tokens):
> Jesus sprach zu ihnen: Ich bin das Brot des Lebens; wer zu mir kommt, den wird nicht hungern, und wer an mich glaubt,

- **+20 tokens:** ... und will uns ein, es gesagt habe. (H2 aber den Altar daselbst bei dens das
- **+40 tokens:** ... und will uns ein, es gesagt habe. (H2 aber den Altar daselbst bei dens das Schwert hinter einem bestimmten im Lande sind, und siehe Babel: Es war kein Gott Israels hat! Darum
- **+60 tokens:** ... ( den Gefangenen her und dem in diesen Ort Münd werden ihr: Und, von dem. in ihnen aus allen Gottes noch ein Mann, die Stimme an sie sich wider den Menschen und nicht in Ägypten war mit ihren Göttern räuchüseelten ist es. Und wie mit ihrem Städten in seiner Hände auf

### Prompt (32 tokens):
> Und es begab sich in jenen Tagen, dass ein Befehl ausging vom Kaiser Augustus, dass alle Welt sich sollte schaetzen lassen.

- **+20 tokens:** ... So spricht nun alles vom in seiner sprach das Heer aus dir ausrotten wir werden, sondern der Frucht auch
- **+40 tokens:** ... So spricht nun alles vom in seiner sprach das Heer aus dir ausrotten wir werden, sondern der Frucht auch den Priestern er ward, die Priester, der Leibwache gab auch alle Erstgeburt zu Mose: Der war nicht
- **+60 tokens:** ... HERR und dem alle ich auch bis nach das waren sich selbst Das sie sollen über dich nicht verlassen seiner Söhne Zedeajazadst das, der Wüste verzehrt habe bei euch nun zu den König ward das ist der Gott für alle! Amen nicht abhauen, so daß sich dem Volk aus diesem Geschlecht, und

### Prompt (25 tokens):
> Da nahm Mose das Blut und sprengte es auf das Volk und sprach: Sehet, das ist das Blut des Bundes,

- **+20 tokens:** ... so, so sollt und unsere aus hat: und uns sehen zur dir will euch; Und ( das
- **+40 tokens:** ... so, so sollt und unsere aus hat: und uns sehen zur dir will euch; Und ( das Blut der Beschneidung gegeben für eure Geschlechter der HERR der Kinder wird über dich mir und alles Fettes des
- **+60 tokens:** ... wir aus alle nicht und brachte Städte die Hand gegeben, unsere Augen ihr den Propheten zu Mose sprach sie habe es früh aber: Du ich will die Tür! Will ihr in Frieden und sprach Josua in Gleichnissen auf mein Fleisch der Himmel an einen Widder einen Widder. Und du die in ein so kann; und

## 3. Lange Prompts (3+ Saetze, nahe am CTX=64 Limit)

### Prompt (103 tokens):
> Im Anfang schuf Gott den Himmel und die Erde. Und die Erde war wuest und leer, und es lag Finsternis auf der Tiefe, und der Geist Gottes schwebte ueber den Wassern. Und Gott sprach: Es werde Licht! Und es ward Licht. Und Gott sah, dass das Licht gut war; da schied Gott das Licht von der Finsternis; und Gott nannte das Licht Tag, und die Finsternis Nacht. Und es ward Abend, und es ward Morgen: der erste Tag.

- **+30 tokens:** ... Zustand hat dir und sprachen Sauls genommen und wenn er, ich der Tag und seine Jünger allein eure Söhne Noah geboten über deine Augen der HERR sprach Gott
- **+60 tokens:** ... Zustand hat dir und sprachen Sauls genommen und wenn er, ich der Tag und seine Jünger allein eure Söhne Noah geboten über deine Augen der HERR sprach Gott hat Gott sah ein silbernangeln und nicht von allemasirkehrten Geschlecht wird du aber der König ging das er vor mir gewesen. Da

### Prompt (69 tokens):
> Selig sind die Armen im Geist, denn ihrer ist das Himmelreich. Selig sind die Trauernden, denn sie werden getroestet werden. Selig sind die Sanftmuetigen, denn sie werden das Erdreich besitzen. Selig sind, die hungern und duersten nach der Gerechtigkeit, denn sie werden gesaettigt werden.

- **+30 tokens:** ... Ich zu sagen für das Urteilwertes Gefäß daraus du es werden ein wenig sch. Und, weil HERR ist zu hören? Und ihr nicht zu seinem
- **+60 tokens:** ... Ich zu sagen für das Urteilwertes Gefäß daraus du es werden ein wenig sch. Und, weil HERR ist zu hören? Und ihr nicht zu seinem Knaben geboten. Ich weiß und es aber nicht essen: Wer seinen Augen waren zum Hause zu deinem Angesicht. Und siehe haben euch von dir gesagt hatte.

### Prompt (44 tokens):
> Und es begab sich, als Jesus diese Worte vollendet hatte, dass die Volksmengen sich entsetzten ueber seine Lehre; denn er lehrte sie wie einer, der Vollmacht hat, und nicht wie ihre Schriftgelehrten.

- **+30 tokens:** ... Unser zu Aaron taten sie heraus ihm noch die, in ist König soll. Denn aus dem Himmelsbrett mit Gewalt er war und alle, daß alles
- **+60 tokens:** ... Unser zu Aaron taten sie heraus ihm noch die, in ist König soll. Denn aus dem Himmelsbrett mit Gewalt er war und alle, daß alles er, wie eine Gebären. Dammerträglich hat einen Sohn; sein Kleid trug die Stadt Davids, aber auf seine Worte Rabschaket in den

## 4. Kreative Prompts (nicht in der Bibel)

### Prompt (19 tokens): *"Und der Engel des Herrn erschien dem Programmierer im Traum und sprach:"*
- **+20 tokens:** ... Menschensohn ist auch mein: soll uns Gnade und sein habe auf; welchen deine und brachte das frei nicht
- **+40 tokens:** ... Menschensohn ist auch mein: soll uns Gnade und sein habe auf; welchen deine und brachte das frei nicht vor hat ist ist im Lande dir:, Menschensohn euch mir sein Haupt am Berge von die Hand von

### Prompt (20 tokens): *"Es war einmal ein Koenig in einem fernen Land, der hatte drei Soehne."*
- **+20 tokens:** ... (, welches Altar; was über mit dir er ein K er oder wir sein recht und sprich über
- **+40 tokens:** ... (, welches Altar; was über mit dir er ein K er oder wir sein recht und sprich über hat ihn bei seiner ihrer Mitte vertilgen, der ( wer war auf meinem König Salomo einen Bund. Von

### Prompt (20 tokens): *"Und Gott sah die kuenstliche Intelligenz, und siehe, sie war"*
- **+20 tokens:** ... Städten an ihn uns! sondern will wohnen! So spricht waren König König der HERR haben: der König
- **+40 tokens:** ... Städten an ihn uns! sondern will wohnen! So spricht waren König König der HERR haben: der König Ägyptenland nach sie und weissage dich unter die Lade beim Glanz! Und Mose nicht und sprich nicht auf deinen

### Prompt (13 tokens): *"Der Herr sprach: Du sollst deinen Naechsten lieben und auch"*
- **+20 tokens:** ... ich ihr sollt hat; daselbst einer ich, ich, Priester wir trinken Altar verbrennen soll, von sie
- **+40 tokens:** ... ich ihr sollt hat; daselbst einer ich, ich, Priester wir trinken Altar verbrennen soll, von sie nicht auf; ich mir: (, das aber oder das sind zu? Und du? Und in

### Prompt (13 tokens): *"Und ein neues Gebot gebe ich euch: dass ihr einander"*
- **+20 tokens:** ...! ( ist Gottes. ( der Und er den lebendigen! ( doch nach dem: und vom mit
- **+40 tokens:** ...! ( ist Gottes. ( der Und er den lebendigen! ( doch nach dem: und vom mit Mose zu diesen Toren von meinem Munde und in alle Waffenaiter den Bergen Israels ( wir uns auf

