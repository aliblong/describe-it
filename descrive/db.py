import logging
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy_utils import database_exists, create_database

from descrive.db_models import Subject, Listing, SubjectListing


def select_subject(subject, sess):
    return sess.query(Subject).filter_by(name=subject).first()


def update_date_scraped(subject, sess):
    subject_entry = select_subject(subject, sess)
    now = datetime.utcnow()
    subject_entry.date_last_scraped = now
    sess.merge(subject_entry)
    sess.commit()


def probe_subject(subject, sess):
    '''
    Returns (id, ads_to_skip), where id is the primary key for use in the
    subject_listings table, and ads_to_skip is the list of ads already in the
    DB, or None if the subject has been scraped recently (< 1 day)
    '''
    subject_entry = select_subject(subject, sess)
    if not subject_entry:
        new_subject = Subject(subject)
        sess.add(new_subject)
        sess.commit()
        sess.refresh(new_subject)
        subject_id = new_subject.id
    else:
        subject_id = subject_entry.id
        now = datetime.utcnow()
        time_since_last_scrape = now - subject_entry.date_last_scraped
        freshly_scraped = time_since_last_scrape.days < 7
        if freshly_scraped:
            return subject_id, None

    listings = sess.query(Listing.id)
    # each returned element from a single-column select is a single-element
    # tuple
    return subject_id, set([listing[0] for listing in listings])


def write_ads_to_db(subject_id, ads, sess):
    logging.info('Writing new ads to DB')
    listings = [Listing(id, **ad) for id, ad in ads.items()]
    for listing in listings:
        sess.merge(listing)
        sess.merge(SubjectListing(subject_id, listing.id))
        sess.commit()


def write_subject_listing_relations_for_already_scraped_ads_to_db(
        subject_id,
        already_scraped_ad_ids,
        sess
):
    logging.info('Updating subject with ads already scraped')
    for already_scraped_ad_id in already_scraped_ad_ids:
        sess.merge(SubjectListing(subject_id, already_scraped_ad_id))
        sess.commit()


def connect_db(url):
    '''
    Returns a db session
    '''
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    return Session()


def delete_dupes(sess):
    sess.execute('''
    delete from listings a using listings b
        where a.id < b.id and a.description = b.description;
    ''')
