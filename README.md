TL;DR: I just want to run it.

```sh
cargo run --release --bin othello
```
Use up/down arrow keys and enter to select player.  Use mouse to make moves.

# Othello Game in Rust: A Journey With Cursor

I was able to implement this in about a day using a language (Rust) with which I have a beginner's level of knowledge.  The intent of this is to demonstrate what can be done with Cursor
in about a day to 1.5 days effort. [^1]

Why Othello?  A long time ago, in a galaxy far, far away… I was a TA (Teacher’s Assistant) for a
101-level C++ programming class.  The class project was to make an Othello game playable with two
human players sharing the same terminal.  The TAs in the class built the game.  It took me about a day to
accomplish that.  After building it, I was bored, so I decided to implement the recursive min-max algorithm with
a simplistic scoring system that used weighted values for the board positions.  Corners and edges were worth more,
spaces immediately in front of them were worth less, and everything else had a baseline value.  Because of this,
I have both domain knowledge and prior experience in implementing such a system.

Why Rust?  Because it is stupid fast (especially when compared to runtimes like Node or Java) yet still has memory safety.
The superior speed will allow it to implement more intensive algorithms with deeper lookaheads
and richer evaluation functions.  Rust’s safety will allow me to focus on the game itself and rely on the
language itself for safety.  In fact, Rust takes safety to sometimes painful levels.  Example: IEEE floating point is only partially ordered.  Why? Because `NaN` is not equal to anything, not even itself.  So you must decide what to do with `NaN`.
Options include allowing the system to Panic (exit ungracefully) if encountered (better be sure it doesn’t happen),
or to handle it as a separate condition.  What you can’t do is ignore it: you will get a compiler error.  It almost goes
without saying: there is no such thing as a magic `null` or `undefined` value that can be used with all types.
The closest thing is the `Option` enum type, which is more like Java’s `Optional` type than a null: Option values may
be `None` or `Some(value)`. [^2] Rust avoids the [Billion Dollar Mistake](https://www.youtube.com/watch?v=ybrQvs4x0Ps) of
null references and implicit behavior in general.

Why Cursor? Because I wanted to see if and how I could develop something in a domain I know, using a language that I can
read but would have difficulty writing it in.

After a few false starts, the implementation strategy I found that worked best for this kind of project is to request a
design or implementation document, review it, make changes (or ask Cursor to make changes to it).  Once the document is
sound, then in a new session request Cursor to implement the design or implementation document.  The “new session” is important.
By breaking apart the design / implementation design, and implementation execution into separate steps, token exhaustion is
avoided.  Other benefits are the ability to review and amend the design before it’s executed, and having the documentation itself
for reference.  One of the ways you know Rust is not a human developer is that it actually updates the documentation to reflect changes that
are requested when such changes touch on parts of the codebase covered by the documentation.

The first type of Othello opponent I requested is a [Monte Carlo Tree Search (MCTS)](MCTS_DESIGN.md).  I requested this because
the programs that beat the world’s best players are often written using this algorithm.  Unfortunately, they also run on some
of the world’s biggest computers, which allow them a large sample size with a correspondingly good confidence rating.
My humble laptop and desktop computers processing, feeble in comparison to the mighty machines that defeat champions, are easily
beaten.  Even allowing for a multi-threaded MCTS algorithm and up to 60 seconds to process per move resulted in a weak player.

But… This was not project failure.  Rather, it illustrates a strength of AI-assisted coding: My design failed, but I can quickly implement
another design.  So I ask Rust to implement min-max A/B pruning [^3], the algorithm I did in C++ a long time ago in a galaxy far, far away.
Once that was working, I asked Rust to parallelize the algorithm based on the available processors.  To say min-max A/B pruning did better than MCTS is an understatement.  If you think you are good at Rust, the Minmax (expert) may make you question your life choices.

#### A typical game with Minmax (expert) kicking my butt.

![Min max beating me (black) 16 to 48 (white)](/readme-images/MinMaxExpertWin.png "Me (black) defeated 16 to 48")

If you have new a 16+ core / 32+ CPU thread machine, the “Expert” mode will be even more merciless because it will be less
likely to finish before it is completed due to the five-second timeout.

I left the MCTS in as a reference so you can see the difference in the performance.  Normally, MCTS would be removed.
You can set both the white and black to computer algorithms to compare them.

## Takeaways:

* The generate documentation / review and amend / execute documentation workflow works well and minimizes surprises.  You also have up-to-date documentation for what you built.
* Use feature branches for new features. That way, if things go off the rails, it is easy to get back to the good code
that is in the main or master branch.  In a corporate setting, this is likely how it is already done.
* It’s OK to try different approaches.  The rapid prototyping capability of AI-assisted development [^4] makes choosing
the wrong approach not a big deal.  The flip side is that a better approach may be found via this prototyping
that would be too risky to try with conventional development methods.
* The ugly: Using the Cursor IDE instead of my favorite IntelliJ.  Cursor does weird things.  It randomly reformats all of my files with whitespace (which would undoubtedly cause immense joy to the people reviewing such a PR - Pull Request).  Even merging branches worked weirdly.  It staged the changes as if I had made them in the target branch and had me commit them.  I resorted to the git command line merge because I do not understand what Cursor is doing, and do not wish to lose the "branch merge" parent commit that a normal merge operation would create.
* AI will not replace you, but a developer fluent in AI tools may.  This technology has well documented security issues.  It hallucinates (makes stuff up).  Yet with all this said, the productivity improvements are too compelling to be ignored.

The cursor generated documentation for the game itself may be found in [ABOUT-GAME.md](ABOUT-GAME.md).

[^1]: Yes, the commit timestamps span days, but this is because this was worked on during vacation during unfocused “free time”.  My guesstimate is that it represents 1 - 1.5 days of focused work time.

[^2]: Enums in Rust are different than Java.  In Java, the enum itself defines a type with certain inherited methods and zero or more methods defined in the enum itself that all the singleton enum values implement or inherit from the base enum definition.  In Rust, an enum itself is more like a collection of related types.  Enum values
can be bound to additional values (such as Some, which is bound with its value).  Such enum values are obviously not singletons.

[^3]: The A/B pruning sounds fancy, but it is simple: Once the game finds a winning move (that is, a move that for all possible opponent moves a countermove exists that leads to a winning state), it stops traversing the move tree for any sibling
moves and returns that move.  Once you have a winning move, evaluating any other moves is pointless.

[^4]: I refuse to say “vibe coding”.  You need to know your problem domain and be able to at least read the language being generated.  You need to review what has been generated at each stage and assume there will be mistakes.  This includes reviewing all code changes using your VCS (Version Control System) before committing them.  If this had been a professional coding project, I would not have relied only on Cursor’s unit tests, but
I would have written my own unit tests for the chosen min-max A/B pruning implementation.  AI tools are not a magic Genie to be summoned to grant the user’s wishes.  Rather, they are a force multiplier for developers to produce code more rapidly in partnership with the AI.