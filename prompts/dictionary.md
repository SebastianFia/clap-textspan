# Auditory-description dictionary: complete single-request specification

You are designing a general-purpose auditory vocabulary for a scientific study of the frozen contrastive language–audio model CLAP. This vocabulary will be embedded by CLAP's text encoder and used to explain variation in attention-head contributions with the TextSpan algorithm. The goal is a broad, readable set of candidate sound concepts. You are NOT captioning an audio clip, interpreting a particular head, inferring an attention pattern, assigning benchmark labels, or optimizing a measured result.

Generate the COMPLETE collection of 800 short English auditory descriptions in this ONE response. Plan the category coverage internally, write the collection, and check it before emitting the final structured object. You do not need any tools, web access, files, audio, model activations, or additional user messages. All necessary instructions and examples are below. Do not request more context.

## Deliverable and machine-readable contract

The response must be a JSON object with ONLY the field descriptions, whose value is an array of exactly 800 objects. Every description object has ONLY these four string fields:

{"id":"audio-v2-0001","category":"nature_animals","text":"A dog barking repeatedly.","provenance_batch":"single-prompt-v1"}

Thus the overall form is {"descriptions":[{...},{...},...]} with all 800 actual entries; the ellipses here only illustrate the structure and must not occur in your output. Do not output code, a generator script, Markdown fences, comments, a prose preface, a downloadable-file claim, a table, or a completion summary. The caller will serialize this structured object into the final dictionary.jsonl file.

Use consecutive IDs audio-v2-0001 through audio-v2-0800, in ascending order. provenance_batch is literally single-prompt-v1 for all entries. The category is metadata; it is not part of the text that CLAP sees. Use normal English spacing and punctuation, with no leading/trailing whitespace or embedded newlines/tabs. Each description should have 3–20 whitespace-separated words, usually 5–12. Choose compact descriptions that fit comfortably in a short text encoder.

## Vocabulary design goals

The bank should cover recognizable sources/events AND abstract auditory properties. The source-oriented half names sounds or actions; the property-oriented half can capture a shared quality across very different sources. Both are useful. Think of a head reacting to a distribution of inputs: the candidate vocabulary should offer several plausible semantic directions without assuming that each clip or head has a single source label.

Favor descriptions of something plausibly audible. A source with a coherent audible action or attribute is useful: a door creaking, a saxophone playing a raspy phrase, a motor pulsing unevenly. Distinguish meaningful events and properties rather than filling quotas with minor adjective, plural, punctuation, or synonym changes. At the same time, related sounds and contrasting acoustic properties are welcome; this is not a taxonomy requiring all concepts to be disjoint. Some template repetition is natural in short sound captions. Aim for varied content rather than ornamental prose.

Represent common everyday sounds, a reasonable range of less common sources, temporally structured events, sustained backgrounds, short transients, and multi-source relationships. Include contrasting poles where meaningful: high/low, smooth/rough, regular/irregular, continuous/intermittent, sharp/soft onset, near/far. Do not make every entry a rare or highly specific event. Use a mix of ordinary specificity and useful distinctions.

Ground details in hearing. Avoid visual facts, exact unseen locations, named people or brands, unsupported causes, inferred personal identity, filenames, dataset names, psychological judgments, numerical pitch/loudness measurements, and implausible adjective combinations. Generic descriptions such as 'An interesting sound' or 'A noise can be heard' contribute little. Naming several alternative sources ('a drill or a vacuum or an engine') is less useful than a coherent event or source-independent texture.

Do not refer to benchmark results, requested target classes, numerical model dimensions, head indices, or expected intervention effects. No source is privileged because it might help a classification task. Do not include negative/positive coefficients, model explanations or claims of experimental success.

## Ten category blocks: exactly 80 entries each

Keep the following order and ID ranges. Categories organize diversity, not the eventual experimental labels.

### nature_animals: Animals and natural environments
Use 0001–0080. Cover mammals, birds, insects and other animal vocalizations; flowing/dripping/splashing water; rain, wind and thunder; fire; vegetation and ground interactions. Include distinct actions and sound textures, not mainly dog/cat variations. Source-specific audible repetition or texture is useful. Do not describe scenery or infer species from generic pitch alone.

### machines_transport: Machines, tools and transportation
Use 0081–0160. Cover engines/motors, varied vehicle operating states, tools, mechanical cycles, electric devices, transport signals, rolling/rattling/friction and industrial mechanisms. Include quiet steady machinery as well as loud transient events. Make source/action relationships coherent, and distinguish operating sounds from merely naming an object.

### household_materials: Everyday objects, actions and materials
Use 0161–0240. Cover doors/windows, utensils, containers, fabric, paper, wood, glass, metal, footsteps and manual object interactions; impacts, scraping, tearing, pouring, washing, rattling and closure/opening actions. Material names should support a plausible audible event. Maintain breadth across rooms/actions/materials without asserting unseen rooms.

### pitch_tonality: Pitch, tonality and spectral organization
Use 0241–0320. Mainly source-independent properties: low/high pitch, stable/unstable tones, rising/falling/gliding pitch, harmonic/dissonant qualities, ringing/drone/whine, narrowband/broadband contrasts, pitch combinations and resonances. A property can be described naturally without naming an instrument or machine.

### voice_vocalization: Human voices and vocal behavior
Use 0321–0400. Cover speech styles and cadence, solitary/group voices, singing and nonverbal vocal sounds, laughter, breathing, coughs, whispers, shouts, overlapping voices, voiced/unvoiced or nasal/raspy qualities. Describe audible behavior instead of guessing identity, nationality, personality or internal feelings. Source properties can be combined when physically plausible.

### music_instruments: Musical sources and performance textures
Use 0401–0480. Cover string, wind, brass, percussion, keyboard and electronic instruments; distinct performance actions, ensemble textures, vocal/instrument combinations, sustained and transient notes, simple musical patterns and dynamics. Include acoustic and electronic sources. Avoid identifying a particular copyrighted tune, artist or composition.

### texture_timbre: Timbre, texture and noise character
Use 0481–0560. Mainly source-independent descriptions of airy/buzzy/raspy/rough/smooth/metallic/woody/hollow/dense/crackling/granular qualities, mixtures of tone and noise, spectral density and resonance. A coherent short texture can contain one or two modifiers. Do not turn these entries into long adjective lists or mostly repeat source names from earlier categories.

### time_rhythm: Temporal pattern and event organization
Use 0561–0640. Mainly source-independent regular/irregular repetition, isolated versus grouped transients, pulse, alternation, accelerating/decelerating sequences, pauses, intermittent activity, stutters, overlapping rhythms and continuous versus discrete activity. Emphasize how events unfold, without requiring a hidden cause or exact numerical tempo.

### dynamics_envelope: Loudness evolution and onset/decay
Use 0641–0720. Mainly source-independent sharp/soft attacks, short/long decays, gradual swelling, abrupt stopping, sustained levels, fades, bursts, amplitude modulation, changing intensity and envelope contrasts. Describe relative loudness and shape rather than invented decibel measurements. Keep the temporal-envelope focus distinct from event identity.

### space_background_relations: Auditory space, scenes and source relationships
Use 0721–0800. Audible distance, dryness/reverberation/echo, foreground/background, dominant/subdued sources, overlapping/masking, apparent approach/recession and coherent mixtures. Include several source-independent relations and several concrete source mixtures. Spatial descriptions must be supportable by hearing; avoid exact room dimensions, visually known locations and unsupported microphone geometry.

## Concrete style and coverage examples

The following examples illustrate suitable language, specificity and acoustic distinctions for each category. They are not additional output fields or additional category quotas. Use them to understand the target style and the range of useful sound concepts. Your final answer still consists of the 800 entries in the specified category blocks.

### Examples: nature_animals
- A dog barking repeatedly.
- A cat meowing repeatedly.
- A horse snorting forcefully.
- A pig grunting repeatedly.
- Geese honking in a group.
- A gull giving piercing cries.
- Bees buzzing continuously.
- A frog croaking repeatedly.
- Rain pattering on leaves.
- A river rushing strongly.
- Wind rustling through leaves.
- Gravel shifting under footsteps.
- A fire crackling steadily.
- Water gurgling inside a cave.
- A squirrel chattering rapidly.
- A bat making rapid squeaks.

### Examples: machines_transport
- A car engine idling steadily.
- Tires squealing during a turn.
- A motorcycle engine rattling loudly.
- A bicycle freewheel ticking rapidly.
- A train clattering over tracks.
- A subway rumbling underground.
- A boat motor puttering softly.
- A motor slowing to a stop.
- A drill boring into wood.
- A grinder scraping metal.
- A printer feeding paper.
- An electronic device emitting beeps.
- A pump throbbing at intervals.
- A ventilation duct rattling.
- A lawn mower running steadily.
- A crossing signal beeping rhythmically.

### Examples: household_materials
- A wooden door creaking open.
- A drawer sliding on rails.
- Footsteps splashing through puddles.
- A table vibrating after impact.
- Dishes clattering in a sink.
- A glass clinking against another.
- A kettle whistling loudly.
- A microwave timer beeping.
- Water pouring into a glass.
- A shower spraying water.
- Paper tearing in a strip.
- Tape peeling from a roll.
- Cloth rubbing against cloth.
- A plastic bag rustling.
- A ball bouncing repeatedly.
- A clock ticking evenly.

### Examples: pitch_tonality
- A low-pitched continuous hum.
- A thin piercing whistle.
- A tone rising gradually in pitch.
- A pitch sliding down then up.
- Two tones sounding in unison.
- Several clashing pitched tones.
- A short upward pitch chirp.
- A three-note repeating signal.
- A narrow band of high frequencies.
- A sound dominated by bass.
- A stable harmonic drone.
- A noisy sound becoming tonal.
- An ascending siren wail.
- A sequence of identical pitches.
- A voice rising in pitch.
- A cluster of closely spaced tones.

### Examples: voice_vocalization
- One person speaking clearly.
- A voice speaking very softly.
- A voice with a low register.
- A voice with a gravelly quality.
- Rapidly spoken syllables.
- Speech with clear consonants.
- A person laughing openly.
- A person sobbing audibly.
- A baby babbling syllables.
- A person sighing slowly.
- A person coughing repeatedly.
- A person swallowing audibly.
- A person humming a melody.
- A voice singing without words.
- A person whistling a tune.
- A distorted voice over a radio.

### Examples: music_instruments
- A piano playing single notes.
- An electric piano playing softly.
- An acoustic guitar being strummed.
- A bass guitar playing a riff.
- A violin playing a sustained note.
- A double bass being plucked.
- A flute playing a clear melody.
- A saxophone playing a raspy phrase.
- A trumpet playing a bright phrase.
- A brass ensemble playing chords.
- A snare drum rolling rapidly.
- A hi-hat opening and closing.
- Bongos playing alternating strokes.
- A xylophone playing bright notes.
- A synthesizer producing a sustained pad.
- A small band playing together.

### Examples: texture_timbre
- A smooth continuous sound.
- A brittle crackling sound.
- A bright metallic rattle.
- A dry papery rustle.
- A breathy airy sound.
- A crackly electrical texture.
- A sound with a sharp edge.
- A thin tinny sound.
- A metallic sound ringing freely.
- A ceramic sound clinking sharply.
- A fluttering noisy texture.
- A popping coarse-grained texture.
- A sound with audible clipping.
- A sound with a shimmering overtone.
- A watery plopping sound.
- A low grinding friction sound.

### Examples: time_rhythm
- A single isolated sound.
- A sequence of brief sounds.
- A regular ticking pattern.
- A repeated triple pulse.
- A rhythm speeding up gradually.
- A rhythm with uneven accents.
- A sound starting and stopping repeatedly.
- A sequence ending with a final hit.
- A sound fluttering rapidly.
- A sound cycling through several textures.
- A fast trill of notes.
- A sparse series of large impacts.
- A steady beat in groups of two.
- A rhythm restarting after a pause.
- A sustained sound lasting several seconds.
- A repeating sound fading between pulses.

### Examples: dynamics_envelope
- A very quiet sound.
- A sound with constant loudness.
- A sound with a sharp attack.
- A sound decaying slowly.
- A quiet sound followed by a loud one.
- A sudden surge in volume.
- A percussive sound with no sustain.
- A sound with repeated sharp peaks.
- A sound with slow loudness fluctuations.
- A sound emerging from silence.
- A sharp impact followed by ringing.
- A bowed note sustaining evenly.
- A recording with large volume contrasts.
- A series of impacts with varying strength.
- A tone fading while noise remains.
- A sound trailing into a faint vibration.

### Examples: space_background_relations
- A sound with strong reverberation.
- A sound with diffuse reflections.
- A voice echoing in a large room.
- A sound with a boxy room resonance.
- A sound moving from left to right.
- A sound circling in the stereo field.
- A narrow mono-like sound image.
- A faint foreground masked by noise.
- Speech over continuous machinery.
- Footsteps over a crowd murmur.
- Two similar sources sounding together.
- A background event becoming the foreground.
- Several layers dropping out one by one.
- A busy passage between quiet intervals.
- A sound partly blocked and muffled.
- A sound followed by the same sound in echo.

## Planning and final consistency check

Before finalizing, ensure each 80-entry block covers multiple subthemes rather than concentrating on one source or wording pattern. Check that the property categories really contain useful properties, that the source categories name audible events/actions, and that mixtures are coherent. Natural related concepts are acceptable; exact case-insensitive duplicate description texts anywhere in the 800-entry collection are not.

Check: 800 entries; IDs 0001–0800 once each in order; the exact ten category strings in their 80-entry blocks; four required string fields; constant provenance_batch; nonempty 3–20-word texts; ordinary spaces; no embedded newlines; no extra fields. All metadata must be literal as specified. Do not add a narrative audit or claim that the bank has been experimentally validated. Return only the complete structured object.
