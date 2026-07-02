from src.training.early_stopping import EarlyStopping

early = EarlyStopping()

losses = [

    1.00,

    0.90,

    0.85,

    0.84,

    0.84,

    0.85,

    0.86,

    0.87

]

for epoch, loss in enumerate(losses, start=1):

    stop = early.should_stop(loss)

    print(

        f"Epoch {epoch} | "

        f"Loss={loss:.2f} | "

        f"Stop={stop}"

    )

    if stop:

        print()

        print(

            f"Training stopped at "

            f"Epoch {epoch}"

        )

        break