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
    title: 'Simple & Unified',
    imageSrc: '/img/puzzle.png',
    description: (
      <>
        One consistent API for all models: <code>fit(model, data, options)</code>.
        From simple linear regression to complex photosynthesis models,
        no need to learn different interfaces.
      </>
    ),
  },
  {
    title: 'Built-in Plotting',
    imageSrc: '/img/flame.png',
    description: (
      <>
        Automatic visualization adapts to your model type.
        1D models show fit curves, multi-dimensional models show response surfaces,
        and photosynthesis models get comprehensive plots with 3D surfaces.
      </>
    ),
  },
  {
    title: 'Comprehensive',
    imageSrc: '/img/plant.png',
    description: (
      <>
        Unified framework spanning generic curve fitting, hydraulics, and photosynthesis.
        9 generic models, 2 hydraulics models, and specialized physiological models
        all accessible through the same simple interface.
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
