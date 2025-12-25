use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use othello::*;
use othello::mcts::MCTSPlayer;

fn bench_mcts_different_iterations(c: &mut Criterion) {
    let game = Game::new();
    
    let mut group = c.benchmark_group("mcts_iterations");
    
    for iterations in [10, 50, 100, 200, 500].iter() {
        let player = MCTSPlayer::with_iterations("Bench", *iterations);
        group.bench_with_input(
            BenchmarkId::from_parameter(iterations),
            iterations,
            |b, _| {
                b.iter(|| {
                    player.choose_move(black_box(&game));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_mcts_with_heuristics(c: &mut Criterion) {
    let game = Game::new();
    
    let mut group = c.benchmark_group("mcts_heuristics");
    
    // Without heuristics
    let player_no_heuristics = MCTSPlayer::with_iterations("Bench No Heuristics", 100);
    group.bench_function("no_heuristics", |b| {
        b.iter(|| {
            player_no_heuristics.choose_move(black_box(&game));
        });
    });
    
    // With heuristics
    let mut player_with_heuristics = MCTSPlayer::with_iterations("Bench With Heuristics", 100);
    player_with_heuristics.set_use_heuristics(true);
    group.bench_function("with_heuristics", |b| {
        b.iter(|| {
            player_with_heuristics.choose_move(black_box(&game));
        });
    });
    
    group.finish();
}

fn bench_mcts_time_limit(c: &mut Criterion) {
    let game = Game::new();
    
    let mut group = c.benchmark_group("mcts_time_limit");
    
    // Without time limit
    let player_no_limit = MCTSPlayer::with_iterations("Bench No Limit", 1000);
    group.bench_function("no_time_limit", |b| {
        b.iter(|| {
            player_no_limit.choose_move(black_box(&game));
        });
    });
    
    // With time limit
    let mut player_with_limit = MCTSPlayer::with_iterations("Bench With Limit", 10000);
    player_with_limit.set_max_time_ms(Some(100));
    group.bench_function("100ms_limit", |b| {
        b.iter(|| {
            player_with_limit.choose_move(black_box(&game));
        });
    });
    
    group.finish();
}

fn bench_node_expansion(c: &mut Criterion) {
    use othello::mcts::node::MCTSNode;
    
    let game = Game::new();
    
    c.bench_function("node_expansion", |b| {
        b.iter(|| {
            let mut node = MCTSNode::new(black_box(game.clone()));
            node.expand();
        });
    });
}

criterion_group!(
    benches,
    bench_mcts_different_iterations,
    bench_mcts_with_heuristics,
    bench_mcts_time_limit,
    bench_node_expansion
);
criterion_main!(benches);

