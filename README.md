#### TL;DR: I just want to run it.

```sh
cargo run --release --bin othello
```
Use up/down arrow keys and enter to select player.  Use the mouse to make moves.  The game documentation is in [ABOUT-GAME.md](ABOUT-GAME.md).  Install Rust from [rustup.rs](https://rustup.rs/).

# Othello Game in Rust: A Journey With Cursor

I was able to implement this in about a day and a half [^1] using a language (Rust) with which I have a beginner's level of knowledge.  I used the Cursor Pro plan trial period.  The intent of this is to see how long it takes to implement a working Othello game using AI tools such as Cursor. 

Why Othello?  A long time ago, in a galaxy far, far away… I was a Teacher’s Assistant for a
101-level C++ programming class.  The class project was to make an Othello game playable with two
human players sharing the same terminal.  The TAs in the class built the game.  It took me about a day to
accomplish that.  After building it, I was bored, so I decided to implement the recursive min-max A/B pruning [^2] algorithm with
a simplistic scoring system that used weighted values for the board positions.  That took about a week to implement.  Because of this,
I have both domain knowledge and prior experience in implementing such a system.

Why Rust?  Because it is stupid fast (especially when compared to runtimes like Node or Java) yet still has memory safety.
The superior speed allows it to implement more intensive algorithms with deeper lookaheads
and richer evaluation functions.  Rust’s safety will allow me to focus on the game and rely on the
language for safety.  In fact, Rust takes safety to sometimes painful levels.  Example: IEEE floating point is only partially ordered.  Why? Because `NaN` is not equal to anything, not even itself.  So you must decide what to do with `NaN`.  It almost goes without saying: there is no such thing as a magic `null` or `undefined` value
that can be used with all types.  The closest thing is the `Option` enum type, which is more like Java’s
`Optional` type than a null: Option values may
be `None` or `Some(value)`. [^3] Rust avoids the [Billion Dollar Mistake](https://www.youtube.com/watch?v=ybrQvs4x0Ps) of
null references and implicit behavior in general.  A focus of the Rust language is [preventing programs from entering an unsafe state](https://rustfoundation.org/media/unsafe-rust-in-the-wild-notes-on-the-current-state-of-unsafe-rust/).

Why Cursor? Because it has a free trial period. I wanted to see if and how I could develop something in a domain I know, using a language that I can
read but would have difficulty writing it in.

After a few false starts, the implementation strategy I found that worked best for this kind of project is to request a
design or implementation document, review it, make changes (or ask Cursor to make changes to it).  Once the document is
sound, then in a new session request Cursor to implement the design or implementation document.  The “new session” is important.
By breaking apart the design and implementation execution into separate steps, [token exhaustion](https://www.getmonetizely.com/articles/token-fatigue-why-ai-users-are-tired-of-thinking-in-tokens) is
avoided.  Other benefits are the ability to review and update the design before it’s executed, and having the documentation
for reference.

The first type of Othello opponent I requested is a [Monte Carlo Tree Search (MCTS)](MCTS_DESIGN.md).  I requested this because
the programs that beat the world’s best players are often written using this algorithm.  Unfortunately, they also run on some
of the world’s biggest computers, which allow them a large sample size with a correspondingly good confidence rating.
My humble laptop and desktop computers processing, feeble in comparison to the mighty machines that defeat champions, are easily
beaten.  Even allowing for a multi-threaded MCTS algorithm and up to 60 seconds to process per move resulted in a weak player.

But… This was not project failure.  Rather, it illustrates a strength of AI-assisted coding: My design failed, but I can quickly implement
another design.  I asked Rust to implement min-max A/B pruning, the algorithm I did in C++ a long time ago in a galaxy far, far away.  Reviewing the documentation is important.  The generated code examples in the document used 1000 for a winning state and -1000 for a losing state.  This was a bad design decision.  Any algorithm that evaluated a board outside of those bounds would erratically choose bad moves because it would treat them as winning or losing.  I instructed Cursor to use `f64::INFINITY` for winning, and `f64::NEG_INFINITY` for losing and it updated the document accordingly.
There was also an issue with mobility (how many moves are available) scoring that I fixed in the documentation.
With the corrected document, Cursor was able to generate a min-max A/B pruning player.
Once that was working, I asked Rust to parallelize the algorithm based on the available processors.  To say min-max A/B pruning did better than MCTS is an understatement.  If you think you are good at Othello, the Minmax (expert) may make you question your life choices.

#### A typical game with Minmax (expert) destroying me.

![Min max beating me (black) 16 to 48 (white)](/readme-images/MinMaxExpertWin.png "Me (black) defeated 16 to 48")

If you have new machine with lots of cores / processor threads, the “Expert” mode will be even more merciless. It will be less
likely to place a move before the algorithm finishes due to the 5 second timeout.

I left the MCTS in as a reference so you can see the difference in the performance.  You can set both the white and black to computer algorithms to compare them.  Normally, MCTS would be removed.

## Takeaways:

* The iterative [Spec-anchored development](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html) workflow [^4] works well and minimizes surprises.  You also have up-to-date documentation for what you built.   
* Use feature branches for new features. That way, if things go off the rails, it is easy to get back to the good code
that is in the main or master branch.  In a corporate setting, this is likely how it is already done.
* The Good: It’s OK to try different approaches.  The rapid prototyping capability of AI-assisted development makes choosing
the wrong approach not a big deal.  The flip side is that a better approach may be found via this prototyping
that would be too risky to try with conventional development methods.  Development time in general is reduced.
* The Bad: "Vibe coding."  Don't do it.  You need to have a knowledge of coding and algorithms.  You need to know your problem domain and be able to at least read and make rudimentary changes in the language being generated.  You need to review what has been generated at each stage and assume there will be mistakes. "Distrust and verify" should be your mindset when using these tools.  This includes reviewing all code changes using your VCS (Version Control System) before committing them and again when merging them. You should write your own unit tests (which I would have done if this had been a professional project).  If you use a design or implementation document, review that what is generated matches the document.  AI coding tools are not a magic Genie to be summoned to grant the user’s wishes.  Rather, they are a force multiplier for developers to produce code more rapidly in partnership with the AI.
* The Ugly: Using the Cursor IDE instead of my favorite IntelliJ.  The Cursor VS Code bundle does weird things.  It randomly reformats all of my files with whitespace (which would undoubtedly cause immense joy to the people reviewing such a Pull Request).  Even merging branches worked weirdly.  It staged the changes as if I had made them in the target branch and had me commit them.  I resorted to the git command line merge because I do not understand what Cursor is doing, and do not wish to lose the "branch merge" parent commit that a normal merge operation would create.
* A Prediction: AI will not replace you, but a developer fluent in AI tools may.  This technology has well documented security issues.  It hallucinates (makes stuff up).  Yet with all this said, the prototyping possibilities and productivity improvements are too compelling to be ignored.

The cursor generated documentation for the game may be found in [ABOUT-GAME.md](ABOUT-GAME.md).

[^1]: Yes, the commit timestamps span days, but this is because this was worked on during vacation during unfocused “free time”.  My guesstimate is that it represents 1 - 1.5 days of focused work time.

[^2]: The A/B pruning sounds fancy, but it is simple: Once the game finds a winning move (that is, a move that for all possible opponent moves a countermove exists that leads to a winning state), it stops traversing the move tree for any sibling moves and returns that move.  Once you have a winning move, evaluating any other moves is pointless.

[^3]: Enums in Rust are different than Java.  In Java, the enum defines a type with certain implicitly inherited methods from the `Enum` class and zero or more methods defined in the enum that the singleton enum values implement or inherit from the base enum definition.  In Rust, an enum is more like a collection of related types.  Enum values
can be bound to additional values (such as Some, which is bound with its value).  Such enum values are obviously not singletons.

[^4]: Each change was done by first creating a specification for it and then executing it.  These specifications became part of the version controlled source.  The AI used the previous specifications as references when making a new one and when executing it.  Because the code was always generated by the specifications, and the specifications are kept and treated as source code, this is a specification anchored (Spec-anchored) project.

## License: [Apache-2.0](LICENSE.md)
