import { TestBed } from '@angular/core/testing';

import { Emotion } from './emotion';

describe('Emotion', () => {
  let service: Emotion;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Emotion);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
