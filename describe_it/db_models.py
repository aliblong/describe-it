from sqlalchemy import Column, Integer, BigInteger, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import FetchedValue


Base = declarative_base()


class Listing(Base):
    __tablename__ = 'listings'

    def __init__(self, id, **kwargs):
        self.id = id
        self.set_attributes(kwargs)

    def set_attributes(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    id = Column(BigInteger, primary_key=True)
    title = Column(String)
    img = Column(String)
    url = Column(String)
    details = Column(String)
    description = Column(String)
    date_posted = Column(DateTime)
    location = Column(String)
    price = Column(String)
    date_scraped = Column(DateTime, FetchedValue())


class Subject(Base):
    __tablename__ = 'subjects'

    def __init__(self, name):
        self.name = name

    id = Column(BigInteger, primary_key=True)
    name = Column(String)
    date_last_scraped = Column(DateTime, FetchedValue())


class SubjectListing(Base):
    __tablename__ = 'subject_listings'

    def __init__(self, subject_id, listing_id):
        self.subject_id = subject_id
        self.listing_id = listing_id

    subject_id = Column(BigInteger, primary_key=True)
    listing_id = Column(BigInteger, primary_key=True)
