import torch

encoded_outfits_file = "encoded_outfits_1678397869.1234586.pt"


def main():
    encoded_outfits = torch.load(encoded_outfits_file)

    for outfit in encoded_outfits:
        encoded_tensor = encoded_outfits[outfit]
        ##encoded_tensor.to('cpu').numpy()


if __name__ == "__main__":
    main()