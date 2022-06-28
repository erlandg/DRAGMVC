from sklearn import semi_supervised
from config.defaults import (
    Experiment,
    CNN,
    MLP,
    GraphAttention,
    DDC,
    Fusion,
    Loss,
    Dataset,
    Optimizer,
    GraphMVC,
    DRAGMVC,
)


gmvc = Experiment(
    dataset_config=Dataset(name="graph_dataset"),
    model_config=GraphMVC(
        backbone_configs=(
            CNN(
                input_size = (1, 256, 256),
                layers = [
                    ("conv", 11, 11, 16, None, ('stride', 4)),
                    ("bn",),
                    ("relu",),
                    ("pool", 2, 2),
                    ("conv", 5, 5, 32, "relu", ('padding', 2), ('stride', 2)),
                    ("conv", 5, 5, 32, None, ('padding', 2), ('stride', 2)),
                    ("bn",),
                    ("relu",),
                    ("pool", 2, 2),
                    ("conv", 3, 3, 64, "relu", ('padding', 1)),
                    ("conv", 3, 3, 64, "relu", ('padding', 1)),
                    ("conv", 3, 3, 64, None, ('padding', 1)),
                    ("bn",),
                    ("relu",),
                    ("fc", 256),
                ],
            ),
            MLP(
                input_size = (486,),
                layers = [512, 256],
                use_bn = True
            ),
            MLP(
                input_size = (240,),
                layers = [512, 256],
                use_bn = True
            ),
        ),
        graph_attention_configs = GraphAttention(
            layers = [128],
        ),
        projector_config=None,
        fusion_config=Fusion(method="weighted_mean", n_views=3),
        cm_config=DDC(n_clusters=2),
        loss_config=Loss(
            funcs="ddc_1|ddc_2|ddc_3|reconstruction",
            # Additional loss parameters go here
            epsilon_features=1,
            epsilon_structure=1,
        ),
        optimizer_config=Optimizer(
            learning_rate=1e-4,
            # Additional optimizer parameters go here
        ),
        shared_weights = True,
    ),
    n_epochs=200,
    n_runs=7,
    batch_size=100
)

gmvc_contrast = Experiment(
    dataset_config = Dataset(

        # cora 2708 x (1433,), (2708,) - 7 classes
        # citeseer 3327 x (3703,), (3327,) - 6 classes
        # pubmed 19717 x (500,), (19717,) - 3 classes
        
        name = "cora",
        # normalise_images = True,
        # eval_sample_proportion = .20,
    ),
    model_config = DRAGMVC(
        backbone_configs = (
            # CNN(
            #     input_size = (3, 256, 256),
            #     # pretrained_model = "alexnet",
            #     # pretrained_features_out = 512,
            #     layers = [
            #         ('conv', 11, 11, 8, None, ('stride', 4)),
            #         ('bn',),
            #         ('relu',),
            #         ('pool', 2, 2),
            #         ('conv', 5, 5, 16, 'relu', ('padding', 2), ('stride', 2)),
            #         ('conv', 5, 5, 16, None, ('padding', 2), ('stride', 2)),
            #         ('bn',),
            #         ('relu',),
            #         ('pool', 2, 2),
            #         ('conv', 3, 3, 32, 'relu', ('padding', 1)),
            #         ('conv', 3, 3, 32, 'relu', ('padding', 1)),
            #         ('conv', 3, 3, 32, None, ('padding', 1)),
            #         ('bn',),
            #         ('relu',),
            #         ('fc', 512)
            #     ],
            # ),
            MLP(
                input_size = (1433,),
                layers = [512],
                activation = None,
                use_bn = False,
            ),
            MLP(
                input_size = (2708,),
                layers = [512],
                activation = None,
                use_bn = False,
            ),
        ),
        graph_attention_configs = GraphAttention(
            standard_gcn = True,
            layers = [512, 512, 512, 512],
            activation = "relu",
            use_bn = True,
            skip_connection = False,
            use_bias = True,
            mask = True,
            mask_weighting = True,
            mask_power = 1,
            attention_features = 1,
            # dropout = .5,
        ),
        projector_config = None,
        fusion_config = Fusion(method = "weighted_mean", n_views = 2),
        cm_config = DDC(
            n_clusters = 7,
            layer_type = "graph_conv"
        ),
        loss_config = Loss(
            funcs = "ddc_1|ddc_2|ddc_3|contrast",
            # n_semi_supervised = 500,
            # semi_supervised_equal_label_split = True,
            # semi_supervised_weight = 1.,
            delta = 6.,
            # negative_samples_ratio = .50,
            epsilon_features = .1,
            epsilon_structure = .2,
        ),
        optimizer_config = Optimizer(
            learning_rate = 1e-3
            # Additional optimizer parameters go here
        ),
        shared_weights = True,
        # warmup_epochs = 20,
        warmup_funcs = "contrast",
        warmup_optimizer = Optimizer(
            learning_rate = 1e-3
        ),
    ),
    best_loss_term = "tot",
    n_epochs = 50,
    n_runs = 75,
    batch_size = 2708,
    graphsaint_steps = 4,
    eval_interval = 5,
)
