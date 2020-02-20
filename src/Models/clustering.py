import umap
import matplotlib
matplotlib.use('Agg')
import pylab

def umap_visualization(vae, data):

    X = []
    target = []

    for batch in data:
       _, z, _, _ = vae.forward(batch['image'].float())

       for vector,label in zip(z, batch['group']):

            X.append(vector.cpu().detach().numpy())

            if label == 'PD':
                target.append(1)
            else:
                target.append(0)

    X = np.array(X)

    reducer = umap.UMAP(random_state=42)
    reducer.fit(X)
    X_embedded = reducer.transform(X)

    pylab.scatter(X_embedded[:,0], X_embedded[:,1], c=target)
    pylab.show()
    pylab.savefig('umap.png')

