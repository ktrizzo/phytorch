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
        Flexible, composable physiological model components that can be combined
        to simulate complex plant processes. Build custom workflows from photosynthesis
        to whole-plant hydraulics.
      </>
    ),
  },
  {
    title: 'Built on PyTorch',
    imageSrc: '/img/flame.png',
    description: (
      <>
        Leverage GPU acceleration and automatic differentiation for fast, efficient
        model fitting. Seamlessly integrate with the PyTorch ecosystem for ML-enhanced
        plant modeling.
      </>
    ),
  },
  {
    title: 'Comprehensive',
    imageSrc: '/img/plant.png',
    description: (
      <>
        Models for photosynthesis (FvCB), stomatal conductance (BMF, MED, BWB, BBL),
        plant hydraulics, radiative properties (PROSPECT), and environmental responses
        all in one toolkit.
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
