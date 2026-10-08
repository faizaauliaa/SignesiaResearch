import numpy as np
import torch

from gcn_autoencoder_holdout import (
    create_graph_dataset,
    GCNEncoderDecoder,
    DEVICE
)


MODEL_PATH = (
    "research/gcn/"
    "gcn_autoencoder_holdout.pt"
)


VIDEOS = [
    "a",
    "abjad",
    "acak",
    "adab",
    "adaptasi"
]


OUTPUT_EMBEDDING = (
    "research/gcn/"
    "all_gcn_embeddings.npy"
)

OUTPUT_LABEL = (
    "research/gcn/"
    "all_gcn_labels.npy"
)



def main():

    print(
        "GENERATE ALL VIDEO EMBEDDINGS"
    )


    model = GCNEncoderDecoder()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )


    model.to(
        DEVICE
    )

    model.eval()


    all_embeddings = []
    all_labels = []


    with torch.no_grad():

        for video in VIDEOS:

            print(
                "Memproses:",
                video
            )


            graphs = create_graph_dataset(
                video
            )


            for graph in graphs:

                graph = graph.to(
                    DEVICE
                )

                batch = torch.zeros(
                    graph.x.size(0),
                    dtype=torch.long,
                    device=DEVICE
                )


                embedding, _ = model(
                    graph.x,
                    graph.edge_index,
                    torch.tensor([0], device=DEVICE)
                )


                all_embeddings.append(
                    embedding.cpu().numpy().squeeze()
                )


                all_labels.append(
                    video
                )


    embeddings = np.array(
        all_embeddings
    )

    labels = np.array(
        all_labels
    )


    np.save(
        OUTPUT_EMBEDDING,
        embeddings
    )

    np.save(
        OUTPUT_LABEL,
        labels
    )


    print()

    print(
        "SELESAI"
    )

    print(
        "Embedding:",
        embeddings.shape
    )

    print(
        "Label:",
        labels.shape
    )



if __name__ == "__main__":

    main()