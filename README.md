PIML Supply-Chain Disruption: Simple Explanation
=================================================

What is this project?
----------------------

There's a research paper that tries to solve this problem: a company
can see its own orders and deliveries, but not what's happening at its
supplier. Can you tell, just from your own numbers, WHEN a supplier had
a problem and for how long? The paper calls this method "PIML" -- a
small neural network that's taught some physics rules about how a
backlog builds up, so its guesses stay realistic.

The paper's own code and data aren't public (available "on request"
only), so we couldn't just run their experiment. Instead we rebuilt it
ourselves, tested it, and then pushed further than the paper did.

What we actually did, step by step
-----------------------------------

1. We built our own simulator. It generates fake supplier "episodes"
   that follow the same physics rules as the paper (orders coming in,
   deliveries going out, backlog piling up when a supplier is in
   trouble, recovering afterward).

2. We rebuilt the paper's model (PIML) and tested it against three
   other methods:
   - A simple curve-fit (no neural network, just solves the physics
     equations directly for each case)
   - Random Forest (a common, simple machine learning model)
   - LSTM (a different kind of neural network, often used for
     sequences)

3. We compared all four on 1,000 simulated test cases. The surprising
   result: the plain curve-fit and Random Forest both beat the paper's
   PIML model. We did not just accept this -- we checked it wasn't a
   mistake (checked the code for cheating/leaks, looked at individual
   examples by eye, checked the result held across 5 different random
   setups). It wasn't a bug. It's a known effect in machine learning:
   when a problem is simple and clean, solving it fresh each time can
   beat a network trained once to guess it quickly.

4. We then improved the training of PIML (fixed two real issues with
   how it was being trained) and that helped -- it got about 25%
   better -- but it still didn't beat the simple curve-fit.

5. Finally, we found a real public dataset (not made up) -- 180,000+
   real e-commerce orders from a real company, 2015-2017, including
   order dates, delivery dates, and whether each delivery was late. We
   built the same kind of "orders in, deliveries out, backlog" picture
   from this real data and tried our models on it.

   Result: the models didn't find a clean "disruption window" in the
   real data, and we figured out why -- real shipping speed moves
   up and down with order volume every week, while the paper's model
   assumes shipping speed only changes during a rare supplier problem.
   Also, in this dataset deliveries are late about 55% of the time
   almost constantly, not because of one specific event. So the
   paper's model is built for a different kind of problem (one clean,
   isolated supplier breakdown) than what this everyday retail data
   shows (constant, spread-out lateness). That's a genuine, useful
   finding, not a failure.

The headline result, in one line
----------------------------------

On our simulated data, simply solving the physics equations directly
(no neural network) beat the paper's neural network by a wide margin,
and that finding held up to real scrutiny. On real-world data, neither
approach found a believable answer, because the real data doesn't
match the kind of problem the paper's model was built to solve.

Where to look
--------------

- `src/` -- all the code (simulator, models, the script that runs the
  comparison)
- `results/` -- the actual numbers from our comparison, plus full logs
- `real_data/` -- the real-dataset experiment and what we found

What's left to do
-------------------

- Try adding noise and overlapping disruptions to our simulator, to see
  if the neural network starts winning once the problem gets messier
  (closer to the paper's claimed strength)
- Try a real dataset that has one specific, known disruption event
  (like a named shortage or a shipping-lane shutdown), instead of
  everyday retail lateness
- Explore whether an LLM/RAG or multi-agent approach could help --
  not yet attempted

Honesty notes
--------------

This is our own rebuild of the paper's method based on reading its
equations, not the authors' actual code, so nothing here is a claim
about bugs in their real implementation. All of the "simulated" data
is ours, not theirs. We haven't tuned PIML's training as much as the
paper says they did.
