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
    title: 'Robust Parameter Estimation',
    imageSrc: '/img/puzzle.png',
    description: (
      <>
        Extract physiological parameters from gas exchange data with confidence.
        Fit FvCB photosynthesis, stomatal conductance, and PROSPECT optical models
        with automatic validation and constraint enforcement.
      </>
    ),
  },
  {
    title: 'GPU-Accelerated Fitting',
    imageSrc: '/img/flame.png',
    description: (
      <>
        Leverage PyTorch's automatic differentiation and GPU acceleration to fit
        hundreds of A-Ci curves simultaneously. Handle complex models with 10+ parameters
        efficiently and robustly.
      </>
    ),
  },
  {
    title: 'From LI-COR to Parameters',
    imageSrc: '/img/plant.png',
    description: (
      <>
        Load your LI-COR 6800 data and extract Vcmax, Jmax, stomatal sensitivity (g₁),
        leaf optical properties, and more. Production-ready tools for plant physiology
        research and breeding programs.
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
