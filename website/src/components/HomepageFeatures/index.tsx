import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  imageSrc: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Modular',
    imageSrc: '/img/puzzle.png',
    description: (
      <>
        Flexible, composable models that allow for customization and experimentation.
        Easily extend with your own models, submodels, and constraints.
      </>
    ),
  },
  {
    title: 'Built on PyTorch',
    imageSrc: '/img/flame.png',
    description: (
      <>
        Leverage GPU acceleration and automatic differentiation for fast, efficient
        parameter estimation. Fit hundreds of curves simultaneously with complex models
        (10+ parameters) using state-of-the-art optimization.
      </>
    ),
  },
  {
    title: 'Comprehensive',
    imageSrc: '/img/plant.png',
    description: (
      <>
        Unified framework for extracting model parameters from data across domains of leaf gas exchange, hydraulics, and optics all in one toolkit.
      </>
    ),
  },
];

function Feature({title, imageSrc, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img
          src={imageSrc}
          alt={title}
          style={{
            height: '100px',
            width: '100px',
            marginBottom: '1rem',
            objectFit: 'contain'
          }}
        />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
