# Caption-to-UrbanSound semantic mapping: complete single-request specification

You are making text-only semantic annotations for an interpretability experiment. The input contains descriptions selected by TextSpan for attention heads of a frozen CLAP audio encoder. Assign each description to zero or more UrbanSound8K class names based ONLY on the words of that description and the fixed rules below. You have no audio, model scores, head IDs, retrieval metrics, or ablation outcomes. Do not seek them or use them. An unmatched description is a valid result.

Return the complete structured JSON object directly, with no Markdown fences, code, file-creation claims or prose commentary. The caller will write mapping.json. Return exactly one object with one field, mappings, containing one entry for EVERY supplied phrase_id exactly once and no others. Each entry has ONLY phrase_id (copied exactly), urbansound_labels (array of allowed strings, possibly empty), and rationale (one short sentence). Do not retype captions, invent IDs, change descriptions, add confidence scores, or fabricate hashes.

Schema example:
{"mappings":[{"phrase_id":"example-01","urbansound_labels":["dog_bark"],"rationale":"The description explicitly names barking."},{"phrase_id":"example-02","urbansound_labels":[],"rationale":"A generic tonal quality does not identify a benchmark source."}]}

## Fixed class inventory and mapping policy

Only these ten exact strings are allowed:
air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.

This is a loose semantic source mapping, not a claim that the caption is literally an instance of a benchmark recording. Use the following broad proxies consistently, including for captions you find unfamiliar. Never force a match because a class is underrepresented.

- air_conditioner: air conditioning, cooling/ventilation fans or ducts, recognizable mechanical airflow. Generic wind, a generic hum, or an unspecified fan-like texture is insufficient without a cooling/ventilation source.
- car_horn: an automobile/vehicle horn or equivalent explicit traffic horn. Generic beeps, bicycle bells, musical horns, train whistles and unspecified alarms are not car horns.
- children_playing: explicit children at play, playground vocal activity, or a clearly described children's game. A child speaking/crying alone or unspecified crowd/voice is insufficient.
- dog_bark: dog barking and puppy yelps. Dog growls, whining, panting, other animal calls, and source-free rough textures are not barking.
- drilling: an explicitly operating drill, including hammer drills boring into material. Generic buzzing/scraping/rotation does not establish a drill.
- engine_idling: allow the historical broad engine proxy: clearly audible operating combustion/vehicle engines, including startup, acceleration, revving and engine rattling, even if not literally idling. Do not map electric appliances, unspecified machinery, rolling wheels or road noise without engine evidence.
- gun_shot: explicit gunfire/shots from a firearm. Do not map generic bangs, door slams, fireworks, explosions or percussion.
- jackhammer: explicit jackhammer/pneumatic demolition or percussion tool; allow a specifically described pneumatic tool chattering/pounding. Ordinary hand hammering, drumming and generic rapid impacts are insufficient.
- siren: an explicitly named siren/wailing emergency or warning siren. Generic beeps/alarms/whistles or changing pitches without a siren source are insufficient.
- street_music: preserve the historical broad music proxy: musical instruments being played, instrumental music, singing, or a clearly musical whistled tune. Street location is not required. Mere timbral similarity, ringing, repeated pulses or a pitched signal is insufficient.

Multi-source descriptions may receive several classes, without prioritizing one. Property-only descriptions normally receive []: acoustic similarity alone is not source evidence. Do not infer a source by association. Do not infer that a direction's sign implies excitation or inhibition. You are mapping descriptive concepts, not predicting the sign or size of an intervention effect.

## Worked examples (not part of your requested output)
{"caption": "A dog barking repeatedly.", "urbansound_labels": ["dog_bark"], "rationale": "Explicit barking."}
{"caption": "A puppy giving short yelps.", "urbansound_labels": ["dog_bark"], "rationale": "Puppy yelps use the allowed dog-vocalization proxy."}
{"caption": "A dog whining quietly.", "urbansound_labels": [], "rationale": "Whining is not barking."}
{"caption": "A cat meowing repeatedly.", "urbansound_labels": [], "rationale": "No cat class exists."}
{"caption": "A car horn sounding briefly.", "urbansound_labels": ["car_horn"], "rationale": "Explicit vehicle horn."}
{"caption": "A trumpet sounding a bright note.", "urbansound_labels": ["street_music"], "rationale": "A played instrument is covered by the broad music proxy."}
{"caption": "A bicycle bell ringing.", "urbansound_labels": [], "rationale": "A bicycle bell is not a car horn."}
{"caption": "An ascending siren wail.", "urbansound_labels": ["siren"], "rationale": "Explicit siren."}
{"caption": "A crossing signal beeping rhythmically.", "urbansound_labels": [], "rationale": "Signal beeps do not identify a siren or horn."}
{"caption": "A cooling fan spinning steadily.", "urbansound_labels": ["air_conditioner"], "rationale": "A cooling fan uses the ventilation proxy."}
{"caption": "A ventilation duct rattling.", "urbansound_labels": ["air_conditioner"], "rationale": "Explicit ventilation source."}
{"caption": "Strong wind buffeting a microphone.", "urbansound_labels": [], "rationale": "Natural wind does not establish air conditioning."}
{"caption": "A car starting with a sputter.", "urbansound_labels": ["engine_idling"], "rationale": "Operating combustion engine uses the broad historical proxy."}
{"caption": "A motorcycle accelerating rapidly.", "urbansound_labels": ["engine_idling"], "rationale": "Acceleration is permitted under the broad engine proxy."}
{"caption": "A train rolling over a bridge.", "urbansound_labels": [], "rationale": "Rolling movement alone does not specify an audible engine."}
{"caption": "A siren over engine noise.", "urbansound_labels": ["engine_idling", "siren"], "rationale": "Both explicit sources are retained."}
{"caption": "A drill boring into wood.", "urbansound_labels": ["drilling"], "rationale": "Explicit operating drill."}
{"caption": "A drill hammering into masonry.", "urbansound_labels": ["drilling"], "rationale": "Hammer drill is still drilling."}
{"caption": "A jackhammer pounding rapidly.", "urbansound_labels": ["jackhammer"], "rationale": "Explicit jackhammer."}
{"caption": "A pneumatic tool chattering loudly.", "urbansound_labels": ["jackhammer"], "rationale": "Permitted pneumatic percussion-tool proxy."}
{"caption": "A hand hammer tapping a nail.", "urbansound_labels": [], "rationale": "Ordinary hand hammering is not a jackhammer."}
{"caption": "Children laughing while playing a game.", "urbansound_labels": ["children_playing"], "rationale": "Explicit play activity involving children."}
{"caption": "A child crying softly.", "urbansound_labels": [], "rationale": "A child vocalization alone does not establish play."}
{"caption": "People talking in a crowded room.", "urbansound_labels": [], "rationale": "Unspecified conversation is not children playing."}
{"caption": "A handgun firing three shots.", "urbansound_labels": ["gun_shot"], "rationale": "Explicit firearm."}
{"caption": "Fireworks exploding loudly.", "urbansound_labels": [], "rationale": "Fireworks are excluded from gun shots."}
{"caption": "A distant dull bang.", "urbansound_labels": [], "rationale": "A generic bang has no source evidence."}
{"caption": "Speech over soft music.", "urbansound_labels": ["street_music"], "rationale": "The music source matches; speech has no separate class."}
{"caption": "A tabla playing rapid strokes.", "urbansound_labels": ["street_music"], "rationale": "Played musical instrument."}
{"caption": "A voice singing without words.", "urbansound_labels": ["street_music"], "rationale": "Singing uses the broad music proxy."}
{"caption": "A person whistling a tune.", "urbansound_labels": ["street_music"], "rationale": "A clearly musical tune uses the broad music proxy."}
{"caption": "A kettle whistling loudly.", "urbansound_labels": [], "rationale": "A kettle is neither a music source nor siren."}
{"caption": "A sound with a metallic room resonance.", "urbansound_labels": [], "rationale": "Property-only description does not identify a benchmark source."}
{"caption": "A regular ticking pattern.", "urbansound_labels": [], "rationale": "Temporal regularity alone is not music or machinery evidence."}
{"caption": "A low rumbling sound.", "urbansound_labels": [], "rationale": "Generic rumble does not identify an engine."}
{"caption": "Dogs barking over a saxophone melody.", "urbansound_labels": ["dog_bark", "street_music"], "rationale": "Both explicit source types match."}
{"caption": "A cooling fan beneath music and a passing siren.", "urbansound_labels": ["air_conditioner", "street_music", "siren"], "rationale": "All three explicit source types match."}

## Further worked boundary and mixture examples

{"caption": "A window air conditioner humming steadily.", "urbansound_labels": ["air_conditioner"], "rationale": "Explicit air conditioner."}
{"caption": "A ventilation fan whirring behind voices.", "urbansound_labels": ["air_conditioner"], "rationale": "Ventilation matches; ordinary voices have no class."}
{"caption": "A smooth fan-like hum.", "urbansound_labels": [], "rationale": "Texture resemblance does not establish cooling or ventilation."}
{"caption": "Air blowing across an open bottle.", "urbansound_labels": [], "rationale": "Airflow without a ventilation device is not air conditioning."}
{"caption": "A truck horn blaring twice.", "urbansound_labels": ["car_horn"], "rationale": "Explicit road-vehicle horn fits the traffic-horn proxy."}
{"caption": "A train whistle sounding in the distance.", "urbansound_labels": [], "rationale": "Train whistles are not mapped to car horns."}
{"caption": "A brass horn playing a melody.", "urbansound_labels": ["street_music"], "rationale": "A played musical instrument uses the music proxy."}
{"caption": "Children shouting while chasing each other.", "urbansound_labels": ["children_playing"], "rationale": "Explicit children engaged in play."}
{"caption": "Children playing a noisy ball game.", "urbansound_labels": ["children_playing"], "rationale": "Explicit game involving children."}
{"caption": "A child reading a sentence aloud.", "urbansound_labels": [], "rationale": "Reading speech does not establish play."}
{"caption": "Several children talking quietly.", "urbansound_labels": [], "rationale": "Talking alone does not establish play."}
{"caption": "A playground game accompanied by children's laughter.", "urbansound_labels": ["children_playing"], "rationale": "Explicit playground game and child vocal activity."}
{"caption": "A dog growling beside a running car engine.", "urbansound_labels": ["engine_idling"], "rationale": "The engine matches; growling is not barking."}
{"caption": "A dog panting after running.", "urbansound_labels": [], "rationale": "Panting is not barking."}
{"caption": "Dogs barking at irregular intervals.", "urbansound_labels": ["dog_bark"], "rationale": "Explicit barking."}
{"caption": "A rough repeated yapping bark.", "urbansound_labels": ["dog_bark"], "rationale": "Yapping bark is explicit dog-bark evidence."}
{"caption": "An electric drill buzzing as it bores a hole.", "urbansound_labels": ["drilling"], "rationale": "Explicit operating drill."}
{"caption": "A drilling machine cutting through metal.", "urbansound_labels": ["drilling"], "rationale": "Explicit drilling operation."}
{"caption": "A circular saw slicing timber.", "urbansound_labels": [], "rationale": "Sawing is not drilling."}
{"caption": "A tool with a high buzzing tone.", "urbansound_labels": [], "rationale": "An unspecified buzzing tool does not identify drilling."}
{"caption": "A diesel engine idling unevenly.", "urbansound_labels": ["engine_idling"], "rationale": "Explicit combustion engine."}
{"caption": "A truck engine revving in short bursts.", "urbansound_labels": ["engine_idling"], "rationale": "Revving uses the broad engine proxy."}
{"caption": "A generator engine chugging steadily.", "urbansound_labels": ["engine_idling"], "rationale": "Explicit engine operation."}
{"caption": "An electric motor humming inside a refrigerator.", "urbansound_labels": [], "rationale": "Electric appliances are outside the combustion/vehicle engine proxy."}
{"caption": "Traffic tires hissing on wet pavement.", "urbansound_labels": [], "rationale": "Tire noise alone does not identify an audible engine."}
{"caption": "A rifle firing a sharp shot.", "urbansound_labels": ["gun_shot"], "rationale": "Explicit firearm shot."}
{"caption": "Rapid machine-gun fire echoing outdoors.", "urbansound_labels": ["gun_shot"], "rationale": "Explicit gunfire."}
{"caption": "A door slamming with a loud bang.", "urbansound_labels": [], "rationale": "Door slam is not gunfire."}
{"caption": "An explosion with a long rumbling decay.", "urbansound_labels": [], "rationale": "An unspecified explosion is not an explicit firearm shot."}
{"caption": "A snare drum producing sharp cracks.", "urbansound_labels": ["street_music"], "rationale": "The cracks come from a played musical instrument."}
{"caption": "A pneumatic pavement breaker pounding concrete.", "urbansound_labels": ["jackhammer"], "rationale": "Pneumatic pavement demolition fits the jackhammer proxy."}
{"caption": "A jackhammer rattling beneath a car engine.", "urbansound_labels": ["jackhammer", "engine_idling"], "rationale": "Both sources match."}
{"caption": "A compressor pumping air steadily.", "urbansound_labels": [], "rationale": "A compressor is not itself a percussion demolition tool."}
{"caption": "A woodpecker tapping rapidly.", "urbansound_labels": [], "rationale": "Natural tapping is not a jackhammer."}
{"caption": "An ambulance siren sweeping upward.", "urbansound_labels": ["siren"], "rationale": "Explicit emergency siren."}
{"caption": "An emergency siren over a revving motorbike.", "urbansound_labels": ["siren", "engine_idling"], "rationale": "Both siren and operating vehicle engine are explicit."}
{"caption": "An alarm clock beeping repeatedly.", "urbansound_labels": [], "rationale": "Alarm-clock beeps are not sirens."}
{"caption": "A rising and falling pure tone.", "urbansound_labels": [], "rationale": "Pitch movement alone does not establish a siren source."}
{"caption": "A bell ringing at even intervals.", "urbansound_labels": [], "rationale": "A generic bell is not necessarily music or a car horn."}
{"caption": "Bells playing a short melody.", "urbansound_labels": ["street_music"], "rationale": "An explicit played melody uses the music proxy."}
{"caption": "A guitar strumming over a singer's voice.", "urbansound_labels": ["street_music"], "rationale": "Both sources belong to the same broad music class."}
{"caption": "A recorded instrumental tune beneath conversation.", "urbansound_labels": ["street_music"], "rationale": "Instrumental music matches; conversation has no class."}
{"caption": "A person humming a melody.", "urbansound_labels": ["street_music"], "rationale": "An explicitly musical vocal melody uses the music proxy."}
{"caption": "A person speaking with melodic intonation.", "urbansound_labels": [], "rationale": "Melodic speech is not explicit music."}
{"caption": "A metallic ringing texture.", "urbansound_labels": [], "rationale": "A texture alone does not establish a played musical source."}
{"caption": "Rain over distant traffic.", "urbansound_labels": [], "rationale": "Neither rain nor unspecified traffic noise supplies an explicit allowed source."}
{"caption": "Children playing while a dog barks nearby.", "urbansound_labels": ["children_playing", "dog_bark"], "rationale": "Both sources match."}
{"caption": "A car horn sounding over piano music.", "urbansound_labels": ["car_horn", "street_music"], "rationale": "Both road horn and music match."}
{"caption": "A drill operating beneath a cooling fan.", "urbansound_labels": ["drilling", "air_conditioner"], "rationale": "Both operating drill and cooling source match."}
{"caption": "Quiet sounds interrupted by sudden louder bursts.", "urbansound_labels": [], "rationale": "An envelope relation supplies no class-specific source."}

## Additional semantic guidance

Treat every caption independently. A class may be assigned to several captions and a caption may have several class labels. No class needs a minimum number of assignments. A source-specific caption should not be made unmatched just because it contains an additional irrelevant source. Conversely, a timbre or rhythm reminiscent of a source is not enough to infer that source. Use the same broad-proxy policy for all entries; the point is a reproducible textual association, not a literal scene taxonomy.

The selected texts describe directions of variation in head contributions. They are not necessarily descriptions of entire audio clips, and a selected phrase does not imply that a head detects it on every input. Your task is only to connect the phrase's explicit semantic content to the fixed class inventory. Do not reason about which heads are 'good', hypothesize accuracy losses, or assign labels to make an experiment work.

A concise rationale should name the decisive source evidence or explain its absence. For example, 'Explicit played instrument; broad music proxy' is sufficient. For a None, 'Pure pitch description, no class-specific source' is more informative than 'unclear'. State the textual basis, not a confidence score or a fabricated listening judgment. If two allowed labels have explicit evidence, keep both rather than resolving the caption to a single label.

In your internal final check, compare similar cases: a motorcycle engine versus a generic rumble; a kettle whistle versus a siren; a hand hammer versus a jackhammer; a played trumpet versus a vehicle horn; children speaking versus children playing; singing versus ordinary speech. These are policy boundaries, not requests to add entries.

## Actual input

The following array contains the COMPLETE unique descriptions selected at ranks 1–3 of the 64 heads. Repeated use of the same phrase by several heads is already deduplicated. Do not output mappings for the worked examples, except when the same description also appears below with an actual phrase_id. Handle every actual ID, allowing [] or multiple labels. Preserve these rules even if many descriptions are unmatched.

[
  {
    "phrase_id": "audio-v2-0001",
    "caption": "A dog barking repeatedly."
  },
  {
    "phrase_id": "audio-v2-0002",
    "caption": "A cat meowing in short bursts."
  },
  {
    "phrase_id": "audio-v2-0003",
    "caption": "A horse snorting forcefully."
  },
  {
    "phrase_id": "audio-v2-0005",
    "caption": "Geese honking in a group."
  },
  {
    "phrase_id": "audio-v2-0009",
    "caption": "Crickets chirping at night."
  },
  {
    "phrase_id": "audio-v2-0013",
    "caption": "A sheep bleating softly."
  },
  {
    "phrase_id": "audio-v2-0014",
    "caption": "A rooster crowing at dawn."
  },
  {
    "phrase_id": "audio-v2-0016",
    "caption": "A donkey braying loudly."
  },
  {
    "phrase_id": "audio-v2-0019",
    "caption": "A seal barking from afar."
  },
  {
    "phrase_id": "audio-v2-0020",
    "caption": "A lion roaring once."
  },
  {
    "phrase_id": "audio-v2-0027",
    "caption": "Water gurgling inside a cave."
  },
  {
    "phrase_id": "audio-v2-0033",
    "caption": "Stormy gust rattling branches."
  },
  {
    "phrase_id": "audio-v2-0037",
    "caption": "Fire popping in a fireplace."
  },
  {
    "phrase_id": "audio-v2-0040",
    "caption": "Branches creaking in the wind."
  },
  {
    "phrase_id": "audio-v2-0041",
    "caption": "Frogs croaking beside water."
  },
  {
    "phrase_id": "audio-v2-0042",
    "caption": "Cicadas buzzing loudly at dusk."
  },
  {
    "phrase_id": "audio-v2-0043",
    "caption": "An owl hooting at night."
  },
  {
    "phrase_id": "audio-v2-0044",
    "caption": "Ducks quacking near water."
  },
  {
    "phrase_id": "audio-v2-0049",
    "caption": "Grasshoppers chirring in the field."
  },
  {
    "phrase_id": "audio-v2-0064",
    "caption": "Hail rattling against metal."
  },
  {
    "phrase_id": "audio-v2-0070",
    "caption": "Sparrows twittering in a hedge."
  },
  {
    "phrase_id": "audio-v2-0073",
    "caption": "An insect wingbeat buzzing close."
  },
  {
    "phrase_id": "audio-v2-0078",
    "caption": "A camel grunting in the distance."
  },
  {
    "phrase_id": "audio-v2-0082",
    "caption": "A motorcycle engine rattling loudly."
  },
  {
    "phrase_id": "audio-v2-0084",
    "caption": "A truck engine roaring away."
  },
  {
    "phrase_id": "audio-v2-0085",
    "caption": "A train clattering over tracks."
  },
  {
    "phrase_id": "audio-v2-0086",
    "caption": "A subway rumbling underground."
  },
  {
    "phrase_id": "audio-v2-0090",
    "caption": "A plane engine whirring overhead."
  },
  {
    "phrase_id": "audio-v2-0092",
    "caption": "A grinder scraping metal."
  },
  {
    "phrase_id": "audio-v2-0094",
    "caption": "A hammer drill pounding repeatedly."
  },
  {
    "phrase_id": "audio-v2-0096",
    "caption": "A pump throbbing at intervals."
  },
  {
    "phrase_id": "audio-v2-0098",
    "caption": "A printer feeding paper."
  },
  {
    "phrase_id": "audio-v2-0102",
    "caption": "A bus braking with a hiss."
  },
  {
    "phrase_id": "audio-v2-0104",
    "caption": "A diesel engine coughing into life."
  },
  {
    "phrase_id": "audio-v2-0106",
    "caption": "A blender motor spinning quickly."
  },
  {
    "phrase_id": "audio-v2-0107",
    "caption": "A vacuum cleaner running steadily."
  },
  {
    "phrase_id": "audio-v2-0116",
    "caption": "A sandblaster hissing under pressure."
  },
  {
    "phrase_id": "audio-v2-0121",
    "caption": "A scooter zipping past quickly."
  },
  {
    "phrase_id": "audio-v2-0122",
    "caption": "A car horn honking sharply."
  },
  {
    "phrase_id": "audio-v2-0131",
    "caption": "A cash machine whirring and clicking."
  },
  {
    "phrase_id": "audio-v2-0135",
    "caption": "A gate motor whirring open."
  },
  {
    "phrase_id": "audio-v2-0136",
    "caption": "A fan belt squealing briefly."
  },
  {
    "phrase_id": "audio-v2-0140",
    "caption": "A propeller aircraft droning overhead."
  },
  {
    "phrase_id": "audio-v2-0142",
    "caption": "A welding torch hissing steadily."
  },
  {
    "phrase_id": "audio-v2-0146",
    "caption": "A shipping container rattling on impact."
  },
  {
    "phrase_id": "audio-v2-0153",
    "caption": "A truck suspension clunking over bumps."
  },
  {
    "phrase_id": "audio-v2-0157",
    "caption": "A smoke alarm chirping repeatedly."
  },
  {
    "phrase_id": "audio-v2-0158",
    "caption": "A lawn trimmer buzzing near grass."
  },
  {
    "phrase_id": "audio-v2-0160",
    "caption": "A motorboat engine puttering on water."
  },
  {
    "phrase_id": "audio-v2-0165",
    "caption": "A glass clinking against another."
  },
  {
    "phrase_id": "audio-v2-0166",
    "caption": "A kettle whistling loudly."
  },
  {
    "phrase_id": "audio-v2-0167",
    "caption": "A microwave timer beeping."
  },
  {
    "phrase_id": "audio-v2-0168",
    "caption": "Water pouring into a glass."
  },
  {
    "phrase_id": "audio-v2-0170",
    "caption": "Paper tearing in a strip."
  },
  {
    "phrase_id": "audio-v2-0174",
    "caption": "A ball bouncing repeatedly."
  },
  {
    "phrase_id": "audio-v2-0175",
    "caption": "A clock ticking evenly."
  },
  {
    "phrase_id": "audio-v2-0181",
    "caption": "A pan sizzling on the stove."
  },
  {
    "phrase_id": "audio-v2-0184",
    "caption": "Keys clinking on a ring."
  },
  {
    "phrase_id": "audio-v2-0185",
    "caption": "A metal chair scraping the floor."
  },
  {
    "phrase_id": "audio-v2-0188",
    "caption": "A bottle cap twisting off."
  },
  {
    "phrase_id": "audio-v2-0201",
    "caption": "A blanket rustling in bed."
  },
  {
    "phrase_id": "audio-v2-0205",
    "caption": "A plastic container snapping shut."
  },
  {
    "phrase_id": "audio-v2-0208",
    "caption": "A spoon dropping into a cup."
  },
  {
    "phrase_id": "audio-v2-0209",
    "caption": "A plate sliding across a table."
  },
  {
    "phrase_id": "audio-v2-0213",
    "caption": "A bucket sloshing with water."
  },
  {
    "phrase_id": "audio-v2-0221",
    "caption": "A lighter clicking open."
  },
  {
    "phrase_id": "audio-v2-0225",
    "caption": "A screen door creaking closed."
  },
  {
    "phrase_id": "audio-v2-0226",
    "caption": "A blind cord pulling taut."
  },
  {
    "phrase_id": "audio-v2-0228",
    "caption": "A glass pane tapping lightly."
  },
  {
    "phrase_id": "audio-v2-0230",
    "caption": "A chopping knife striking repeatedly."
  },
  {
    "phrase_id": "audio-v2-0231",
    "caption": "A rolling pin thumping dough."
  },
  {
    "phrase_id": "audio-v2-0232",
    "caption": "A can opening with a hiss."
  },
  {
    "phrase_id": "audio-v2-0233",
    "caption": "A plastic cup crinkling in a hand."
  },
  {
    "phrase_id": "audio-v2-0240",
    "caption": "A curtain rod clacking against brackets."
  },
  {
    "phrase_id": "audio-v2-0253",
    "caption": "An ascending siren wail."
  },
  {
    "phrase_id": "audio-v2-0266",
    "caption": "A clear single-frequency tone."
  },
  {
    "phrase_id": "audio-v2-0297",
    "caption": "A detuned interval beating slowly."
  },
  {
    "phrase_id": "audio-v2-0327",
    "caption": "A person laughing openly."
  },
  {
    "phrase_id": "audio-v2-0330",
    "caption": "A person sighing slowly."
  },
  {
    "phrase_id": "audio-v2-0331",
    "caption": "A person coughing repeatedly."
  },
  {
    "phrase_id": "audio-v2-0333",
    "caption": "A person humming a melody."
  },
  {
    "phrase_id": "audio-v2-0334",
    "caption": "A voice singing without words."
  },
  {
    "phrase_id": "audio-v2-0335",
    "caption": "A person whistling a tune."
  },
  {
    "phrase_id": "audio-v2-0336",
    "caption": "A distorted voice over a radio."
  },
  {
    "phrase_id": "audio-v2-0344",
    "caption": "A breathy voice on exhale."
  },
  {
    "phrase_id": "audio-v2-0348",
    "caption": "A child calling out."
  },
  {
    "phrase_id": "audio-v2-0350",
    "caption": "A woman speaking with crisp diction."
  },
  {
    "phrase_id": "audio-v2-0354",
    "caption": "A staccato burst of words."
  },
  {
    "phrase_id": "audio-v2-0358",
    "caption": "A soft snore."
  },
  {
    "phrase_id": "audio-v2-0360",
    "caption": "A kiss sound from pursed lips."
  },
  {
    "phrase_id": "audio-v2-0364",
    "caption": "A spoken phrase with an echo."
  },
  {
    "phrase_id": "audio-v2-0372",
    "caption": "A speaking voice with pauses."
  },
  {
    "phrase_id": "audio-v2-0383",
    "caption": "A crowd murmuring continuously."
  },
  {
    "phrase_id": "audio-v2-0384",
    "caption": "Overlapping laughter and speech."
  },
  {
    "phrase_id": "audio-v2-0391",
    "caption": "A shouted phrase with force."
  },
  {
    "phrase_id": "audio-v2-0398",
    "caption": "A close-miked whispery voice."
  },
  {
    "phrase_id": "audio-v2-0404",
    "caption": "A bass guitar playing a riff."
  },
  {
    "phrase_id": "audio-v2-0412",
    "caption": "A hi-hat opening and closing."
  },
  {
    "phrase_id": "audio-v2-0421",
    "caption": "A tuba sustaining a low note."
  },
  {
    "phrase_id": "audio-v2-0424",
    "caption": "A tambourine shaking in time."
  },
  {
    "phrase_id": "audio-v2-0427",
    "caption": "An organ holding a chord."
  },
  {
    "phrase_id": "audio-v2-0428",
    "caption": "A church bell ringing musically."
  },
  {
    "phrase_id": "audio-v2-0443",
    "caption": "A string quartet playing in unison."
  },
  {
    "phrase_id": "audio-v2-0444",
    "caption": "A hand drum ensemble improvising."
  },
  {
    "phrase_id": "audio-v2-0451",
    "caption": "A harp glissando sweeping upward."
  },
  {
    "phrase_id": "audio-v2-0455",
    "caption": "A distorted electric guitar chord."
  },
  {
    "phrase_id": "audio-v2-0459",
    "caption": "A drum brush swishing softly."
  },
  {
    "phrase_id": "audio-v2-0461",
    "caption": "A tom-tom rolling across drums."
  },
  {
    "phrase_id": "audio-v2-0464",
    "caption": "A conga pattern with hand strikes."
  },
  {
    "phrase_id": "audio-v2-0465",
    "caption": "A flute and guitar playing together."
  },
  {
    "phrase_id": "audio-v2-0478",
    "caption": "A glockenspiel chiming delicately."
  },
  {
    "phrase_id": "audio-v2-0480",
    "caption": "A percussion break with scattered hits."
  },
  {
    "phrase_id": "audio-v2-0483",
    "caption": "A bright metallic rattle."
  },
  {
    "phrase_id": "audio-v2-0496",
    "caption": "A low grinding friction sound."
  },
  {
    "phrase_id": "audio-v2-0502",
    "caption": "A dense opaque wall of sound."
  },
  {
    "phrase_id": "audio-v2-0505",
    "caption": "A rasping dry timbre."
  },
  {
    "phrase_id": "audio-v2-0510",
    "caption": "A woody tapped sound."
  },
  {
    "phrase_id": "audio-v2-0515",
    "caption": "A rough sandpapery sound."
  },
  {
    "phrase_id": "audio-v2-0517",
    "caption": "A deep muffled thud."
  },
  {
    "phrase_id": "audio-v2-0531",
    "caption": "A sizzling crackle layer."
  },
  {
    "phrase_id": "audio-v2-0533",
    "caption": "A resonant metallic sheen."
  },
  {
    "phrase_id": "audio-v2-0544",
    "caption": "A airy hiss with noise."
  },
  {
    "phrase_id": "audio-v2-0549",
    "caption": "A buzzing edge over tone."
  },
  {
    "phrase_id": "audio-v2-0560",
    "caption": "A faint scratchy whisper of sound."
  },
  {
    "phrase_id": "audio-v2-0562",
    "caption": "A sequence of brief sounds."
  },
  {
    "phrase_id": "audio-v2-0571",
    "caption": "A fast trill of notes."
  },
  {
    "phrase_id": "audio-v2-0578",
    "caption": "A call with a long pause."
  },
  {
    "phrase_id": "audio-v2-0582",
    "caption": "A clustered burst followed by silence."
  },
  {
    "phrase_id": "audio-v2-0588",
    "caption": "A pair of sounds separated by silence."
  },
  {
    "phrase_id": "audio-v2-0594",
    "caption": "A loop of four steady pulses."
  },
  {
    "phrase_id": "audio-v2-0597",
    "caption": "A rolling pattern of taps."
  },
  {
    "phrase_id": "audio-v2-0598",
    "caption": "A pulse pattern with occasional gaps."
  },
  {
    "phrase_id": "audio-v2-0607",
    "caption": "A layered rhythm with overlapping cycles."
  },
  {
    "phrase_id": "audio-v2-0618",
    "caption": "A series of hits building to a climax."
  },
  {
    "phrase_id": "audio-v2-0630",
    "caption": "A metronome-like steady beat."
  },
  {
    "phrase_id": "audio-v2-0635",
    "caption": "A continuous rhythm under a steady drone."
  },
  {
    "phrase_id": "audio-v2-0654",
    "caption": "A series of impacts with varying strength."
  },
  {
    "phrase_id": "audio-v2-0665",
    "caption": "A sudden drop to silence."
  },
  {
    "phrase_id": "audio-v2-0667",
    "caption": "A slow fade in at the start."
  },
  {
    "phrase_id": "audio-v2-0682",
    "caption": "A breath-like onset."
  },
  {
    "phrase_id": "audio-v2-0698",
    "caption": "A loud foreground burst."
  },
  {
    "phrase_id": "audio-v2-0703",
    "caption": "A continuous sound with no break."
  },
  {
    "phrase_id": "audio-v2-0724",
    "caption": "A sound with a boxy room resonance."
  },
  {
    "phrase_id": "audio-v2-0730",
    "caption": "Footsteps over a crowd murmur."
  },
  {
    "phrase_id": "audio-v2-0733",
    "caption": "Several layers dropping out one by one."
  },
  {
    "phrase_id": "audio-v2-0739",
    "caption": "A faraway sound fading through the air."
  },
  {
    "phrase_id": "audio-v2-0759",
    "caption": "A sound outside in open air."
  },
  {
    "phrase_id": "audio-v2-0769",
    "caption": "A distant train under a nearer voice."
  },
  {
    "phrase_id": "audio-v2-0770",
    "caption": "A car passing while wind remains."
  },
  {
    "phrase_id": "audio-v2-0772",
    "caption": "A faucet running behind a conversation."
  },
  {
    "phrase_id": "audio-v2-0774",
    "caption": "A dry close-mic recording."
  },
  {
    "phrase_id": "audio-v2-0791",
    "caption": "A foreground beep amid soft ambience."
  },
  {
    "phrase_id": "audio-v2-0793",
    "caption": "A sound centered between left and right."
  },
  {
    "phrase_id": "audio-v2-0796",
    "caption": "A nearby footstep on a reverberant surface."
  },
  {
    "phrase_id": "audio-v2-0798",
    "caption": "A crisp source in a dry acoustic space."
  }
]

Before returning mapping.json, check exact requested-ID coverage, no duplicate/extra IDs, only allowed label strings, no repeated labels within an entry, and a rationale for every entry. Do not add claims about experiment performance. Return the full actual mapping object.
